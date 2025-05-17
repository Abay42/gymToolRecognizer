from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI
from core.config import settings
from model.user_gym_log import UserGymLog

model = settings.MODEL
client = ChatCompletionsClient(
    endpoint=settings.GITHUB_MODEL_ENDPOINT,
    credential=AzureKeyCredential(settings.GITHUB_TOKEN),
)


def generate_workout(prompt: str, user_logs: list[UserGymLog]):
    log_summary = "\n".join(
        f"{log.date}: {'Посетил' if log.went_to_gym else 'Пропустил'} | {log.action or 'Нет данных'}"
        for log in user_logs
    )

    system_prompt = (
        "Вы — профессиональный AI-тренер по фитнесу. Проанализируйте историю тренировок пользователя, чтобы понять его регулярность, "
        "предпочитаемые упражнения и прогресс. Затем ответьте соответствующим образом."
    )

    response = client.complete(
        model=model,
        messages=[
            SystemMessage(system_prompt + "\n\nИстория тренировок:\n" + log_summary),
            UserMessage(prompt),
        ],
        temperature=1.0,
        top_p=1.0,
    )
    return response.choices[0].message.content.strip()
