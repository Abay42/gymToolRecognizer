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
        f"{log.date}: {'Went' if log.went_to_gym else 'Missed'} | {log.action or 'No details'}"
        for log in user_logs
    )

    system_prompt = (
        "You are a professional AI fitness coach. Analyze the user's training history to understand their consistency, "
        "preferred exercises, and progress. Then respond accordingly."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt + "\n\nTraining history:\n" + log_summary},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )
    return response.choices[0].message.content.strip()


