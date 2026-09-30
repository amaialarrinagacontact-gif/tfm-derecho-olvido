"""Comprueba en 5 segundos si tu clave del modelo de lenguaje funciona.

Uso:  python probar_llm.py
"""
from src import config

if not config.LLM_DISPONIBLE:
    raise SystemExit("No hay clave configurada. Crea el archivo .env con, por ejemplo: MISTRAL_API_KEY=tu_clave")

print(f"Proveedor: {config.LLM_PROVIDER} | modelo: {config.LLM_MODEL}")
clave = config.LLM_API_KEY
print(f"Clave leída del .env: {clave[:4]}...{clave[-3:]} ({len(clave)} caracteres)")

try:
    if config.LLM_PROVIDER == "gemini":
        from google import genai
        r = genai.Client(api_key=clave).models.generate_content(model=config.LLM_MODEL, contents="Di OK")
        respuesta = r.text
    else:
        from openai import OpenAI
        cliente = OpenAI(base_url=config.PROVEEDORES[config.LLM_PROVIDER]["url"], api_key=clave)
        r = cliente.chat.completions.create(model=config.LLM_MODEL, max_tokens=5,
                                            messages=[{"role": "user", "content": "Responde solo: OK"}])
        respuesta = r.choices[0].message.content
    print(f"\nFUNCIONA. Respuesta del modelo: {respuesta!r}")
except Exception as e:
    texto = str(e)
    print(f"\nERROR: {texto[:300]}\n")
    if "401" in texto or "Unauthorized" in texto or "invalid" in texto.lower():
        print("Diagnóstico: la clave no es válida. Cópiala de nuevo (sin espacios) o crea otra.")
    elif "429" in texto or "rate" in texto.lower():
        print("Diagnóstico: la cuenta no tiene cuota disponible.\n"
              " - Si la clave es nueva, espera 5-10 minutos.\n"
              " - En Mistral: activa el plan gratuito 'Experiment' y verifica tu teléfono (Billing / Plans).\n"
              " - Si nada funciona, prueba con Groq: GROQ_API_KEY=... en el .env (y borra la línea de Mistral).")
    elif "404" in texto or "model" in texto.lower():
        print("Diagnóstico: el modelo no existe para tu cuenta. Añade al .env una línea LLM_MODEL=... con otro modelo.")
    else:
        print("Diagnóstico: error de conexión o del proveedor. Comprueba tu internet y vuelve a probar.")
