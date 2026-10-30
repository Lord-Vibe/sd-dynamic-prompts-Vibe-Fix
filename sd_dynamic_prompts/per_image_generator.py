"""Per-image immediate variables for the Vibe-Fix Dynamic Prompts extension.

Adds `${name=!!expression}` without changing `${name=!expression}`.
Only random prompts *containing* `=!!` use the special sampling context;
all other prompts delegate directly to dynamicprompts' original generator.

Based on the public dynamicprompts 0.31.0 parser and sampling-context APIs.
This module never patches the installed dynamicprompts package or Forge Neo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, replace
from uuid import uuid4

from dynamicprompts.commands import Command, SequenceCommand
from dynamicprompts.commands.variable_commands import VariableAssignmentCommand
from dynamicprompts.enums import SamplingMethod
from dynamicprompts.generators.randomprompt import RandomPromptGenerator
from dynamicprompts.samplers.cycle import CyclicalSampler
from dynamicprompts.samplers.random import RandomSampler
from dynamicprompts.sampling_context import SamplingContext
from dynamicprompts.sampling_result import SamplingResult
from dynamicprompts.utils import rotate_and_join


@dataclass
class _ImageFrame:
    number: int = 0


class _ImageBinding(Command):
    """One expression stream with a value cached for the current image."""

    sampling_method = None

    def __init__(self, name, expression_stream, frame):
        self.name = name
        self.expression_stream = iter(expression_stream)
        self.frame = frame
        self.last_image = -1
        self.cached_value = None
        self.resolving = False

    def get_value(self):
        if self.last_image != self.frame.number:
            if self.resolving:
                raise RecursionError(f"Recursive per-image variable: {self.name}")
            self.resolving = True
            try:
                self.cached_value = SamplingResult(
                    text=str(next(self.expression_stream)),
                )
                self.last_image = self.frame.number
            finally:
                self.resolving = False
        return self.cached_value

    def generate(self):
        while True:
            yield self.get_value()


def _mark_immediate_assignments(template: str, variable_start: str, marker: str) -> str:
    """Rewrite `name=!!expr` to `MARKERname=!expr` for the stock parser.

    `!!` is not recognized by the 0.31.0 parser. The marker is confined to the
    *assignment name*; normal `${name}` references are left unchanged and
    linked when the sampling context processes variable assignments.
    """
    pattern = re.compile(
        re.escape(variable_start)
        + r"(?P<space>\s*)(?P<name>[A-Za-z_-][\w-]*)"
        + r"(?P<optional>\s*\??\s*)=!!(?![!])",
    )
    return pattern.sub(
        lambda match: (
            variable_start
            + match.group("space")
            + marker
            + match.group("name")
            + match.group("optional")
            + "=!"
        ),
        template,
    )


@dataclass(frozen=True)
class _ScopedContext(SamplingContext):
    frame: _ImageFrame = field(default_factory=_ImageFrame)
    marker: str = field(default_factory=lambda: "DPPI_" + uuid4().hex + "_")

    def sample_prompts(self, prompt, num_prompts=None):
        if isinstance(prompt, str) and "=!!" in prompt:
            prompt = _mark_immediate_assignments(
                prompt,
                self.parser_config.variable_start,
                self.marker,
            )
        return super().sample_prompts(prompt, num_prompts)

    def process_variable_assignments(self, commands):
        # Mimics upstream's loop, differing only for *marked* assignments.
        bindings = {}
        remaining = []
        for command in commands:
            if not isinstance(command, VariableAssignmentCommand):
                remaining.append(command)
                continue
            is_per_image = command.name.startswith(self.marker)
            if is_per_image:
                command = replace(command, name=command.name[len(self.marker) :])
                if not command.overwrite and command.name in self.variables:
                    value = self.variables[command.name]
                else:
                    value = _ImageBinding(
                        command.name,
                        self.generator_from_command(command.value),
                        self.frame,
                    )
            else:
                value = self.process_variable_assignment(command)
            bindings[command.name] = value
        return remaining, self.with_variables(bindings) if bindings else self

    def generator_from_command(self, command):
        if isinstance(command, _ImageBinding):
            return command.generate()
        return super().generator_from_command(command)


class _ImageSequenceMixin:
    def _get_sequence(self, command: SequenceCommand, context: SamplingContext):
        tokens, scoped_context = context.process_variable_assignments(command.tokens)
        new_bindings = [
            value
            for name, value in scoped_context.variables.items()
            if isinstance(value, _ImageBinding)
            and context.variables.get(name) is not value
        ]
        generators = [scoped_context.generator_from_command(token) for token in tokens]
        while True:
            # One eager evaluation per output, in assignment order. All
            # references inside this output read the same cached value.
            for binding in new_bindings:
                binding.get_value()
            yield rotate_and_join(generators, separator=command.separator)


class _ScopedRandomSampler(_ImageSequenceMixin, RandomSampler):
    pass


class _ScopedCyclicalSampler(_ImageSequenceMixin, CyclicalSampler):
    pass


class ScopeAwareRandomPromptGenerator(RandomPromptGenerator):
    """Retain the stock generator except when `=!!` is requested."""

    def generate(self, template=None, num_images=1, *, seeds=None, **kwargs):
        if not isinstance(template, str) or "=!!" not in template:
            # Most important safety property: do not change existing prompts.
            return super().generate(template, num_images, seeds=seeds, **kwargs)

        if (
            not isinstance(num_images, int)
            or isinstance(num_images, bool)
            or num_images < 0
        ):
            raise ValueError("num_images must be a non-negative integer")
        if num_images == 0:
            return []

        if isinstance(seeds, int):
            seeds = [seeds] * num_images
        elif seeds is not None:
            seeds = list(seeds)
            if len(seeds) == 1:
                seeds *= num_images
            if len(seeds) != num_images:
                raise ValueError(f"Expected {num_images} seeds, got {len(seeds)}")

        base = {
            item.name: getattr(self._context, item.name)
            for item in fields(SamplingContext)
        }
        base["samplers"] = {
            **base["samplers"],
            SamplingMethod.RANDOM: _ScopedRandomSampler(),
            SamplingMethod.CYCLICAL: _ScopedCyclicalSampler(),
        }
        context = _ScopedContext(**base)
        stream = iter(context.sample_prompts(template, num_images))
        expanded = []
        for image_index in range(num_images):
            context.frame.number = image_index
            if seeds is not None:
                context.rand.seed(seeds[image_index])
            expanded.append(str(next(stream)))
        return expanded
