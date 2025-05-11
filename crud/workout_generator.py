from openai import OpenAI
from core.config import settings
from model.user_gym_log import UserGymLog

endpoint = settings.GITHUB_MODEL_ENDPOINT
model = "openai/gpt-4.1"
client = OpenAI(
    base_url=endpoint,
    api_key=settings.GITHUB_TOKEN,
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

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt + "\n\nИстория тренировок:\n" + log_summary},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )
    return response.choices[0].message.content.strip()