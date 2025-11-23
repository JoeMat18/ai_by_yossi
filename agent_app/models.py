from enum import unique
from tabnanny import verbose
from django.db import models
from django.utils.translation import gettext_lazy as _


class ProviderChoices(models.TextChoices):
    OPENAI = "openai", _("OpenAI")
    ANTHROPIC = "anthropic", _("Anthropic")
    GOOGLE = "google_vertexai", _( "Google Gemini")
    HUGGINGFACE = "huggingface", _("Hugging Face")


class PromptType(models.TextChoices):
    INTENT = "intent", _("Intent classifiacation")
    EXTRACT = "extract", _("Paramter extraction")
    GENERAL_QA = "general_qa", _("GENERAL Q&A")


class Prompt(models.Model):
    name = models.CharField(
        max_length=150,
        primary_key=True,
        help_text=_("Unique prompt key")
    )

    type = models.CharField(
        max_length=32,
        choices=PromptType.choices,
        help_text=_("Logical type of this prompt")
    )
    content = models.TextField(
        help_text=_("Prompt text used as a system prompt for the model")
    )

    is_active = models.BooleanField(
        default=True,
        help_text=_("Only active prompts are available for selection")
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = _("Prompt")
        verbose_name_plural = _("Prompts")

    def __str__(self) -> str:
        return f"{self.name} ({self.type})"

class StrategyPromptMapping(models.Model):
    strategy_name = models.CharField(
        max_length=100,
        unique=True,
        help_text=_("Name of the strategy")
    )

    provider = models.CharField(
        max_length=50,
        choices=ProviderChoices.choices,
        help_text=_("Model provider name")
    )

    model_name = models.CharField(
        max_length=100,
        help_text=_("Model identifier")
    )

    # Prompts for different stages of the LangGraph workflow
    intent_prompt = models.ForeignKey(
        Prompt,
        on_delete=models.PROTECT,
        related_name="as_intent_prompt",
        verbose_name=_("Intent Prompt"),
        blank=True,
        null=True,
        limit_choices_to={"type": PromptType.INTENT, "is_active": True},
        help_text=_("System prompt used by the intent-classification node."),
    )

    extract_prompt = models.ForeignKey(
        Prompt,
        on_delete=models.PROTECT,
        related_name="as_extract_prompt",
        verbose_name=_("Extraction Prompt"),
        blank=True,
        null=True,
        limit_choices_to={"type": PromptType.EXTRACT, "is_active": True},
        help_text=_("System prompt used by the parameter-extraction node."),
    )

    general_qa_prompt = models.ForeignKey(
        Prompt,
        on_delete=models.PROTECT,
        related_name="as_general_qa_prompt",
        verbose_name=_("General Q&A Prompt"),
        blank=True,
        null=True,
        limit_choices_to={"type": PromptType.GENERAL_QA, "is_active": True},
        help_text=_("System prompt used by the general Q&A node."),
    )

    is_active = models.BooleanField(
        default=True,
        help_text=_("Only active strategies should be used."),
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


    class Meta:
        ordering = ["strategy_name"]
        verbose_name = _(" Prompts mapping")
        verbose_name_plural = _("Prompts mappings")

    def __str__(self) -> str:
        return self.strategy_name