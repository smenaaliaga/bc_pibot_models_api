# Reglas del normalizer (`app/model/normalizer.py`)

Este documento describe **cómo** y **en qué orden conceptual** se aplican las reglas de normalización de entidades en la API.

---

## 1) Flujo general

El endpoint recibe entidades crudas desde NER (`interpretation.entities`) y ejecuta:

1. Normalización base por entidad (`normalize_ner_entities`).
2. Ajustes API-friendly y reglas de negocio (`normalize_entities`).
3. Resolución final de período (`_resolve_period_value`).

Salida final esperada:

- `indicator`, `seasonality`, `frequency`, `activity`, `region`, `investment`: listas.
- `period`: rango de dos fechas `['YYYY-MM-DD', 'YYYY-MM-DD']`.

---

## 2) Reglas base de matching

### 2.1 Normalización de texto

Antes de comparar:

- minúsculas
- sin acentos
- trim de espacios

Función: `_normalize_text`.

### 2.2 Selección por mejor match fuzzy

Para cada entidad, se evalúa el vocabulario completo y se selecciona la clave con mejor score fuzzy por sobre un umbral.

Funciones: `_best_vocab_key`, `_fuzzy_match`.

### 2.3 Prioridad de negación

Si el texto contiene `no`, se priorizan términos negativos cuando existan en vocabulario.

Ejemplo: `no mineri` -> `no_mineria`.

---

## 3) Reglas de inferencia para `indicator` y `frequency`

Cuando `indicator` es genérico o vacío, la inferencia se aplica como un
**flujo único de validaciones**:

1. **Evaluar `frequency` explícita**:
   - `frequency=m` -> `indicator=imacec`
   - `frequency in (q, a)` -> `indicator=pib`

2. **Si `frequency` está vacía y `req_form=point`, inferir desde `period`**:
   - mes (`junio 2023`) -> `frequency=m` -> `indicator=imacec`
   - trimestre (`primer trimestre 2023`, con typos) -> `frequency=q` -> `indicator=pib`
   - año solo (`2025`) -> `frequency=a` -> `indicator=pib`

3. **Si no se logró inferir por `period` (o `req_form` no es `point`)**:
   - asigna `imacec/m` **solo** si `activity` está en cobertura IMACEC
     y `intents.region` + `intents.investment` son `none`.
    - si hay señales de PIB (`intents.region`/`intents.investment` o
       cobertura de actividades PIB), asigna `pib/q`.

4. **Si no hay señales suficientes**, fallback final: `imacec/m`.

---

## 4) Reglas de `seasonality`

Precedencia:

1. `seasonality` explícita válida (`sa` o `nsa`) prevalece.
2. Si falta o no matchea, se infiere por `calc_mode`:
   - `prev_period` -> `sa`
   - `yoy` -> `nsa`
   - resto (`original`, `contribution`, vacío) -> `nsa`

---

## 5) Reglas de período (`period`)

Función principal: `_resolve_period_value`.

### 5.1 Detección de contexto trimestral

Se considera contexto trimestral si:

- `frequency == 'q'`, o
- el texto de período tiene referencia a trimestre.

La detección de trimestre es resiliente a typos con `_is_trimester_like_token`.

Ejemplo soportado: `primer trismestre 2024`.

### 5.2 Extracción de fechas

Orden de extracción por cada valor de período:

1. Trimestre explícito (si contexto trimestral)
2. Mes + año (o mes con año contextual)
3. Año explícito
4. Fallback `normalize_period`

### 5.3 Formato de salida por `req_form`

- `point`: devuelve `[inicio, fin]` del período detectado.
- `range`: ordena fechas y devuelve `[min_inicio, max_fin]`.
- `latest`: ignora fecha textual y devuelve período anterior al actual según contexto:
  - mensual -> mes anterior
  - trimestral -> trimestre anterior
  - anual (cuando aplica) -> año anterior

### 5.4 Reglas adicionales

- Año solo (`2024`) se expande a `['2024-01-01', '2024-12-31']` en contextos aplicables.
- Rangos invertidos se corrigen automáticamente.
- Si falta año explícito en mes/trimestre, se usa año contextual o año actual.

---

## 6) Casos de ejemplo

### Caso A: economía + trimestre bien escrito

Input:

- `indicator=['economia']`
- `period=['primer trimestre 2024']`

Output esperado:

- `indicator=['pib']`
- `frequency=['q']`
- `period=['2024-01-01', '2024-03-31']`

### Caso B: economía + trimestre con typo

Input:

- `indicator=['economia']`
- `period=['primer trismestre 2024']`

Output esperado (mismo comportamiento):

- `indicator=['pib']`
- `frequency=['q']`
- `period=['2024-01-01', '2024-03-31']`

---

## 7) Dónde están los tests

Revisar `tests/test_normalizer.py`, especialmente casos:

- inferencia con indicador genérico + trimestre
- inferencia con typo en `trimestre`
- snapping trimestral de período
- reglas `latest`, `point`, `range`
