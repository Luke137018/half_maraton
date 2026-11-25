import os
from langfuse import get_client

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APIError,
)


# ============================================================
# 1. KONFIGURACJA APLIKACJI
# ============================================================

st.set_page_config(
    page_title="Prognoza czasu półmaratonu",
    layout="centered",
)

st.title(" Prognoza czasu półmaratonu")

st.write(
    """
    Sprawdź, jaki czas możesz osiągnąć na dystansie półmaratonu.
    Wybierz sposób podania swoich danych.
    """
)


# ============================================================
# 2. ŁADOWANIE MODELU
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "halfmarathon_model.pkl"
)


@st.cache_resource
def load_model():
    """Wczytuje wytrenowany model ML."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# 3. EKSTRAKCJA DANYCH Z OPENAI
# ============================================================


def extract_runner_data(description, api_key):
    """
    Wyciąga z opisu użytkownika:
    - płeć,
    - wiek,
    - czas biegu na 5 km.

    Langfuse rejestruje:
    - nazwę modelu,
    - czas wykonania,
    - zużycie tokenów,
    - błędy API.

    Nie zapisujemy w Langfuse:
    - klucza OpenAI użytkownika,
    - opisu użytkownika,
    - odpowiedzi zawierającej jego dane.
    """

    client = OpenAI(api_key=api_key.strip())

    prompt = f"""
    Przeanalizuj opis biegacza i wyodrębnij następujące informacje:

    1. sex:
       - "M" dla mężczyzny
       - "K" dla kobiety
       - null, jeśli nie można ustalić

    2. age:
       - wiek w pełnych latach jako liczba całkowita
       - null, jeśli nie można ustalić

    3. five_k_time_seconds:
       - czas biegu na 5 km wyrażony w sekundach
       - null, jeśli nie można ustalić

    Czas może być podany jako:
    - "25:30"
    - "00:25:30"
    - "25 minut i 30 sekund"

    Jeśli użytkownik poda rok urodzenia zamiast wieku,
    oblicz wiek na podstawie bieżącego roku 2026.

    Nie zgaduj brakujących danych.

    Zwróć wyłącznie JSON w formacie:

    {{
        "sex": "M",
        "age": 32,
        "five_k_time_seconds": 1530
    }}

    Opis użytkownika:
    {description}
    """

    # Sprawdzenie, czy Langfuse jest skonfigurowany
    langfuse_enabled = (
        bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY"))
    )

    def call_openai():
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Jesteś asystentem wyodrębniającym "
                        "dane biegaczy. Zwracasz wyłącznie JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

    # Jeśli Langfuse nie jest skonfigurowany,
    # aplikacja nadal działa normalnie.
    if not langfuse_enabled:
        response = call_openai()

    else:
        langfuse = get_client()

        try:
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="runner-data-extraction",
                model="gpt-4o-mini",
                model_parameters={
                    "temperature": 0,
                    "response_format": "json_object",
                },
                metadata={
                    "application": "halfmarathon-predictor",
                    "feature": "runner-data-extraction",
                },
            ) as generation:

                try:
                    response = call_openai()

                    # Rejestrujemy wykorzystanie tokenów,
                    # ale nie treść promptu ani odpowiedzi.
                    if response.usage:
                        generation.update(
                            usage_details={
                                "input": response.usage.prompt_tokens,
                                "output": response.usage.completion_tokens,
                                "total": response.usage.total_tokens,
                            }
                        )

                except Exception as e:

                    # Rejestrujemy rodzaj błędu,
                    # bez zapisywania klucza lub treści żądania.
                    generation.update(
                        level="ERROR",
                        status_message=type(e).__name__,
                    )

                    raise

        finally:
            # Wysyłamy oczekujące zdarzenia do Langfuse.
            try:
                langfuse.flush()
            except Exception:
                pass

    result = response.choices[0].message.content

    if not result:
        raise ValueError("OpenAI zwróciło pustą odpowiedź.")

    return json.loads(result)



# ============================================================
# 4. WALIDACJA DANYCH
# ============================================================

def validate_runner_data(data):
    """
    Sprawdza kompletność i poprawność danych.
    """

    missing = []

    sex = data.get("sex")
    age = data.get("age")
    five_k_time = data.get("five_k_time_seconds")

    # Płeć
    if sex not in ["M", "K"]:
        missing.append("płeć (M lub K)")

    # Wiek
    try:
        age = int(age)

        if not 10 <= age <= 100:
            missing.append("wiek od 10 do 100 lat")

    except (TypeError, ValueError):
        missing.append("wiek")

    # Czas na 5 km
    try:
        five_k_time = float(five_k_time)

        if not 600 <= five_k_time <= 5400:
            missing.append(
                "czas biegu na 5 km od 10 do 90 minut"
            )

    except (TypeError, ValueError):
        missing.append("czas biegu na 5 km")

    return missing


# ============================================================
# 5. FORMATOWANIE CZASU
# ============================================================

def format_time(total_seconds):
    """Zamienia sekundy na format HH:MM:SS."""

    total_seconds = int(round(total_seconds))

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ============================================================
# 6. WSPÓLNA FUNKCJA PREDYKCJI
# ============================================================

def predict_halfmarathon_time(runner_data):
    """
    Przyjmuje dane biegacza, waliduje je
    i wykonuje predykcję przy użyciu modelu ML.
    """

    missing = validate_runner_data(runner_data)

    if missing:
        return None, missing

    input_data = pd.DataFrame(
        [
            {
                "Płeć": runner_data["sex"],
                "Wiek": int(runner_data["age"]),
                "5 km Czas": float(
                    runner_data["five_k_time_seconds"]
                ),
            }
        ]
    )

    model = load_model()

    prediction = model.predict(input_data)[0]

    return prediction, []


# ============================================================
# 7. WYBÓR TRYBU
# ============================================================

st.subheader("Jak chcesz podać swoje dane?")

mode = st.radio(
    "Wybierz tryb:",
    options=[
        "Ręcznie – bez OpenAI",
        "Automatycznie – z OpenAI",
    ],
    index=0,
    horizontal=False,
)

runner_data = None
submitted = False


# ============================================================
# 8. TRYB RĘCZNY
# ============================================================

if mode == "Ręcznie – bez OpenAI":

    st.info(
        "Wpisz swoje dane ręcznie. "
        "Ten tryb nie wymaga klucza OpenAI ani połączenia z API."
    )

    with st.form("manual_prediction_form"):

        sex = st.selectbox(
            "Płeć",
            options=["Mężczyzna", "Kobieta"],
        )

        age = st.number_input(
            "Wiek",
            min_value=10,
            max_value=100,
            value=32,
            step=1,
        )

        five_k_minutes = st.number_input(
            "Czas na 5 km – minuty",
            min_value=10,
            max_value=89,
            value=24,
            step=1,
        )

        five_k_seconds = st.number_input(
            "Czas na 5 km – sekundy",
            min_value=0,
            max_value=59,
            value=30,
            step=1,
        )

        submitted = st.form_submit_button(
            "Oszacuj mój czas",
            type="primary",
        )

    if submitted:

        five_k_time_seconds = (
            five_k_minutes * 60 + five_k_seconds
        )

        runner_data = {
            "sex": "M" if sex == "Mężczyzna" else "K",
            "age": int(age),
            "five_k_time_seconds": five_k_time_seconds,
        }


# ============================================================
# 9. TRYB AUTOMATYCZNY – OPENAI
# ============================================================

elif mode == "Automatycznie – z OpenAI":

    st.info(
        "Opisz siebie zwykłym językiem. "
        "OpenAI odczyta z opisu wymagane informacje."
    )

    with st.form("openai_prediction_form"):

        api_key = st.text_input(
            "Twój klucz OpenAI API",
            type="password",
            placeholder="Wklej swój klucz OpenAI API",
            help=(
                "Klucz możesz utworzyć na stronie "
                "https://platform.openai.com/api-keys"
            ),
        )

        description = st.text_area(
            "Opisz siebie",
            placeholder=(
                "Jestem mężczyzną, mam 32 lata. "
                "5 km przebiegam w 24 minuty i 30 sekund."
            ),
            height=150,
        )

        submitted = st.form_submit_button(
            "Oszacuj mój czas",
            type="primary",
        )

    if submitted:

        if not api_key or not api_key.strip():

            st.error("Podaj swój klucz OpenAI API.")

        elif not description or not description.strip():

            st.error("Opisz siebie, aby wykonać prognozę.")

        else:

            try:

                with st.spinner("Analizuję Twój opis..."):

                    runner_data = extract_runner_data(
                        description,
                        api_key,
                    )

            except AuthenticationError:

                st.error(
                    "Nieprawidłowy klucz OpenAI API. "
                    "Sprawdź klucz i spróbuj ponownie."
                )

            except RateLimitError:

                st.error(
                    "Przekroczono limit OpenAI API lub "
                    "wystąpił problem z rozliczeniami konta."
                )

            except APIConnectionError:

                st.error(
                    "Nie udało się połączyć z OpenAI API. "
                    "Spróbuj ponownie."
                )

            except APIError:

                st.error(
                    "Wystąpił błąd OpenAI API. "
                    "Spróbuj ponownie za chwilę."
                )

            except (json.JSONDecodeError, ValueError) as e:

                st.error(
                    f"Nie udało się odczytać danych z opisu: {e}"
                )

            except Exception as e:

                st.error(
                    "Wystąpił nieoczekiwany błąd podczas "
                    "analizy opisu."
                )
                st.exception(e)


# ============================================================
# 10. WSPÓLNA PREDYKCJA I PREZENTACJA WYNIKU
# ============================================================

if submitted and runner_data is not None:

    try:

        with st.spinner("Obliczam przewidywany czas..."):

            predicted_seconds, missing = predict_halfmarathon_time(
                runner_data
            )

        if missing:

            st.warning(
                "Nie udało się uzyskać wszystkich wymaganych danych."
            )

            st.write("Sprawdź następujące informacje:")

            for item in missing:
                st.write(f"- {item}")

        else:

            predicted_time = format_time(predicted_seconds)

            st.success("Gotowe! Udało się oszacować Twój czas.")

            st.subheader(
                "Twój przewidywany czas półmaratonu"
            )

            st.metric(
                label="Prognozowany czas",
                value=predicted_time,
            )

            st.caption(
                "Wynik jest szacunkiem statystycznym opartym "
                "na danych historycznych. Rzeczywisty czas "
                "może zależeć od treningu, pogody, trasy "
                "i innych czynników."
            )

            with st.expander("Dane wykorzystane do prognozy"):

                st.write(
                    f"**Płeć:** {runner_data['sex']}"
                )

                st.write(
                    f"**Wiek:** {runner_data['age']} lat"
                )

                st.write(
                    "**Czas na 5 km:** "
                    f"{format_time(runner_data['five_k_time_seconds'])}"
                )

    except FileNotFoundError as e:

        st.error(str(e))

    except Exception as e:

        st.error(
            "Wystąpił błąd podczas wykonywania predykcji."
        )
        st.exception(e)


# ============================================================
# 11. STOPKA
# ============================================================

st.divider()

st.caption(
    "Projekt edukacyjny – prognozowanie czasu półmaratonu "
    "na podstawie danych historycznych."
)
