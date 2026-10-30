from pathlib import Path
from string import Template
from unittest.mock import patch

import pytest
from dynamicprompts.generators.randomprompt import RandomPromptGenerator
from dynamicprompts.wildcards import WildcardManager

from sd_dynamic_prompts.generator_builder import GeneratorBuilder
from sd_dynamic_prompts.per_image_generator import ScopeAwareRandomPromptGenerator


@pytest.fixture
def manager(tmp_path):
    (tmp_path / "colors.txt").write_text("red\nblue\ngreen\n")
    (tmp_path / "nested.txt").write_text("__colors__\n")
    return WildcardManager(tmp_path)


@pytest.mark.parametrize(
    "prompt",
    [
        "plain text",
        "{red|blue} __colors__",
        "${x=!{red|blue}}${x}/${x}",
        "${x={red|blue}}${x}/${x}",
    ],
)
def test_stock_prompts_unchanged(manager, prompt):
    seeds = list(range(6))
    stock = RandomPromptGenerator(manager, seed=25)
    scoped = ScopeAwareRandomPromptGenerator(manager, seed=25)
    assert scoped.generate(prompt, 6, seeds=seeds) == stock.generate(
        prompt,
        6,
        seeds=seeds,
    )


@pytest.mark.parametrize(
    "expression",
    [
        "{red|blue|green}",
        "__colors__",
        "__nested__",
        "{2::red|blue}",
    ],
)
def test_per_image_references_match(manager, expression):
    generator = ScopeAwareRandomPromptGenerator(manager, seed=25)
    rows = generator.generate(
        "${x=!!" + expression + "}${x}/${x}",
        6,
        seeds=list(range(6)),
    )
    assert len(rows) == 6
    assert all(left == right for left, right in (row.split("/") for row in rows))


def test_mixed_scopes_through_builder(manager):
    generator = GeneratorBuilder(manager).set_seed(25).create_generator()
    assert isinstance(generator, ScopeAwareRandomPromptGenerator)
    rows = generator.generate(
        "${scene=!{@studio|garden}}${x=!!{@red|green|blue}}${scene}/${x}/${x}",
        6,
        seeds=list(range(6)),
    )
    assert rows == ["studio/red/red", "studio/green/green", "studio/blue/blue"] * 2


def test_builder_falls_back_on_import_error(manager):
    with patch.dict("sys.modules", {"sd_dynamic_prompts.per_image_generator": None}):
        generator = GeneratorBuilder(manager).set_seed(25).create_generator()
    assert type(generator) is RandomPromptGenerator
    assert generator.generate("plain text", 2) == ["plain text"] * 2


def test_seed_repeatability(manager):
    template = "${x=!!{red|blue|green}}${x}"

    def generate(seeds):
        return ScopeAwareRandomPromptGenerator(manager, seed=25).generate(
            template,
            6,
            seeds=seeds,
        )

    assert generate(123) == generate([123])
    assert generate(list(range(6))) == generate(list(range(6)))


def test_same_prompt_variable_dependency_is_unsupported(manager):
    generator = ScopeAwareRandomPromptGenerator(manager, seed=25)
    with pytest.raises(KeyError, match="scene"):
        generator.generate("${scene=!{beach|forest}}${x=!!${scene}}${x}", 6)


def test_spaced_operator_does_not_activate_feature(manager):
    template = "${x= !!{red|blue}}${x}"
    scoped = ScopeAwareRandomPromptGenerator(manager, seed=25)
    stock = RandomPromptGenerator(manager, seed=25)
    rows = scoped.generate(template, 6)
    assert rows == stock.generate(template, 6)
    assert all("!!" in row for row in rows)


def test_help_rendering_preserves_examples():
    root = Path(__file__).resolve().parents[2]
    rendered = Template((root / "helptext.html").read_text()).substitute(
        WILDCARD_DIR="/wildcards",
        VERSION="test",
        LIB_VERSION="0.31.0",
    )
    assert "{2$$artist1|artist2|artist3}" in rendered
    assert "${color=!!{red|blue}}" in rendered
    assert "${color= !!{red|blue}}" in rendered
    assert "${scene=!{beach|forest}}${x=!!${scene}}${x}" in rendered


@pytest.mark.parametrize("batch_size,n_iter", [(3, 2), (1, 6)])
def test_extension_expands_six_images(
    monkeypatch_webui,
    processing,
    batch_size,
    n_iter,
):
    from scripts.dynamic_prompting import Script

    processing.batch_size = batch_size
    processing.n_iter = n_iter
    processing.set_prompt_for_test(
        "${scene=!{@studio|garden}}${x=!!{@red|green|blue}}${scene}/${x}/${x}",
    )
    processing.set_negative_prompt_for_test("ugly")
    Script().process(
        p=processing,
        is_enabled=True,
        is_combinatorial=False,
        combinatorial_batches=1,
        is_magic_prompt=False,
        is_feeling_lucky=False,
        is_attention_grabber=False,
        min_attention=0,
        max_attention=1,
        magic_prompt_length=0,
        magic_temp_value=1,
        use_fixed_seed=False,
        unlink_seed_from_prompt=False,
        disable_negative_prompt=False,
        enable_jinja_templates=False,
        no_image_generation=False,
        max_generations=0,
        magic_model="magic",
        magic_blocklist_regex=None,
    )
    assert (
        processing.all_prompts
        == [
            "studio/red/red",
            "studio/green/green",
            "studio/blue/blue",
        ]
        * 2
    )
    assert processing.all_negative_prompts == ["ugly"] * 6
