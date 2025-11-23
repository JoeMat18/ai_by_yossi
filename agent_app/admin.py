from django.contrib import admin

from agent_app.models import Prompt, StrategyPromptMapping

@admin.register(Prompt)
class PromptAdmin(admin.ModelAdmin):
    list_display = ("name", "type", "is_active", "updated_at")
    list_filter = ("type", "is_active")
    search_fields = ("name", "content")

@admin.register(StrategyPromptMapping)
class StrategyPromptMappingAdmin(admin.ModelAdmin):
    list_display = (
        "strategy_name",
        "provider",
        "model_name",
        "is_active",
    )
    list_filter = ("provider", "is_active")
    search_fields = ("strategy_name", "model_name")