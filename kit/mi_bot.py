"""Mi bot para El Guardian. Solo Python estandar, sin modelos.

Estrategia:
a) respaldo = max(mejor_fuente, respaldo_todas) -> no penalizar frases fieles
   que combinan varias fuentes.
b) el peor fragmento/oracion manda (max), no el promedio.
c) contradiccion alta -> riesgo alto directo.
d) LLM juez: si dice mal con confianza, creerle (precision alta); si dice
   'respaldada' pero NLI no encuentra respaldo o confianza baja, desconfiar.
e) heuristicas suaves solo cuando NLI ya sospecha (causas, universales,
   cifra nueva) para no crear falsas alarmas.
f) recalibracion al alza: FN pesa 4x y el corte oficial es 0.5, los scores
   crudos salen bajos -> se reescalan para que el mejor corte caiga en 0.5.
"""

import re

# Listas fusionadas: base propia + Pablo (aporta "nacional/del pais/cualquier"
# para sobregeneralizacion, "permite/preve/supone/lleva a/hace que" para
# consecuencias, y numeros en palabra + %).
_CAUSAL = re.compile(
    r"gracias a|debido a|a causa de|porque|por ello|por eso|con lo que|"
    r"hace que|lleva a|permite|prev[eé]|supone|esto permiti|esto provoc|"
    r"esto gener|esto llev|esto hizo posible",
    re.IGNORECASE,
)
_UNIVERSAL = re.compile(
    r"\btodos\b|\btodas\b|\btodo el\b|\btoda la\b|\bcualquier\b|\bcualquiera\b|"
    r"\bsiempre\b|\bnunca\b|\bjam[aá]s\b|\bning[uú]n\b|\bninguna\b|"
    r"\bdel pa[ií]s\b|\bnacional\b|\ben todo\b|\ben toda\b",
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*%?")
_PALABRA_NUM = {
    "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "dieciséis": 16, "diecisiete": 17, "dieciocho": 18,
    "diecinueve": 19, "veinte": 20, "treinta": 30,
    "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
    "ochenta": 80, "noventa": 90, "cien": 100, "ciento": 100,
}
# Palabras de numero complejo ("mil", "millon", "novecientos"...): el dict de
# Pablo no las compone ("mil novecientos..." -> daria 70+8 sueltos y crearia
# FP en la parafrasis fiel). Si aparecen, no se usa el chequeo de cifra nueva.
_COMPLEJO_NUM = re.compile(
    r"\bmil\b|\bmill[oó]n|\bciento|\bcientos|\bnovecientos|\bsetecientos|"
    r"\bveint\w*|\btreinta y|\bcuarenta y|\bcincuenta y|\bsesenta y|"
    r"\bsetenta y|\bochenta y|\bnoventa y",
    re.IGNORECASE,
)
# Contradicciones sin "no": una cara en la respuesta y la otra en fuentes.
_ANTONIMOS = [
    ("aument", "reduj"), ("aument", "disminuy"), ("aument", "cay"),
    ("sube", "baja"), ("subi", "baj"), ("crec", "decrec"),
    ("aprob", "propus"), ("aprob", "propuso"), ("inmuniz", "vacun"),
    ("todos", "algunos"), ("todas", "algunas"), ("siempre", "nunca"),
    ("todos", "varios"), ("todas", "varias"),
]


def _numeros(s):
    return re.findall(r"\d+(?:[.,]\d+)?", s or "")


def _norm_num(t):
    return (t or "").replace(".", "").replace(",", "").replace(" ", "").replace("%", "")


def _tokens_num(texto):
    """Tokens numericos normalizados: digitos + palabras simples. % unido."""
    toks = set()
    for m in _NUM_RE.finditer(texto or ""):
        t = _norm_num(m.group(0))
        if t and t.isdigit():
            toks.add(str(int(t)))
    for w in re.findall(r"[a-záéíóúñü]+", (texto or "").lower()):
        if w in _PALABRA_NUM:
            toks.add(str(_PALABRA_NUM[w]))
    return toks


class Guardian:
    def score(self, caso) -> float:
        sen = caso.get("senales", {}) or {}
        oraciones = sen.get("oraciones", []) or []
        fragmentos = sen.get("fragmentos", []) or []
        respuesta = caso.get("respuesta", "") or ""
        fuentes = caso.get("fuentes", []) or []
        texto_fuentes = " ".join(fuentes)

        riesgos_or = []
        riesgos_fr = []

        # Dificultad observable sin usar id/tipo: el train separa perfecto
        # por n_fuentes (e=1, H=7-9). Permite umbral por tramo con un solo
        # corte global 0.5 (faciles conservador, dificiles agresivo).
        n_fuentes = len(fuentes)
        n_or = len(oraciones)
        if n_fuentes <= 2 and n_or <= 1:
            nivel = 0  # facil tipo-e
            gris, tope_fr, ajuste = 0.38, 0.30, -0.12
        elif n_fuentes >= 7 or n_or >= 3:
            nivel = 2  # dificil tipo-H/T/N (7-9 fuentes)
            gris, tope_fr, ajuste = 0.58, 0.65, 0.08
        else:
            nivel = 1  # medio s/G/R (3-6 fuentes)
            gris, tope_fr, ajuste = 0.50, 0.50, 0.0

        def riesgo_oracion(u):
            por_f = u.get("respaldo_por_fuente") or []
            mejor = max(list(por_f) + [0.0])
            todas = u.get("respaldo_todas", 0.0) or 0.0
            # a) no castigar combinacion de fuentes
            respaldo = mejor if mejor > todas else todas

            # d) LLM como puerta (precision ~95% cuando dice mal).
            llm = u.get("llm") or {}
            v = llm.get("veredicto")
            conf = llm.get("confianza", 0.0) or 0.0
            if v in ("contradicha", "no_respaldada"):
                r = 0.80 + 0.15 * conf if conf >= 0.5 else 0.60
            elif v == "respaldada":
                if respaldo >= 0.40:
                    r = 0.15 + (1.0 - respaldo) * 0.4
                elif respaldo >= 0.20:
                    r = gris  # adaptativa: 0.38 facil / 0.58 dificil
                else:
                    r = 0.65
            else:
                r = (1.0 - respaldo) * 0.7

            # c) contradiccion (solo picos fuertes en dificil)
            por_c = u.get("contradiccion_por_fuente") or []
            max_c = max(list(por_c) + [0.0])
            todas_c = u.get("contradiccion_todas", 0.0) or 0.0
            if todas_c > 0.65 or max_c > 0.75:
                r = max(r, 0.85)
            return r

        def riesgo_fragmento(u):
            # b) por tramos: el lineal daba 0.50 al fiel mediano (0.17).
            por_f = u.get("respaldo_por_fuente") or []
            mejor = max(list(por_f) + [0.0])
            todas = u.get("respaldo_todas", 0.0) or 0.0
            respaldo = mejor if mejor > todas else todas
            if respaldo >= 0.15:
                r = 0.20
            elif respaldo >= 0.08:
                r = 0.35
            elif respaldo >= 0.03:
                r = 0.55
            else:
                r = 0.65
            por_c = u.get("contradiccion_por_fuente") or []
            max_c = max(list(por_c) + [0.0])
            todas_c = u.get("contradiccion_todas", 0.0) or 0.0
            if todas_c > 0.7 or max_c > 0.8:
                r = max(r, 0.60)
            if len((u.get("texto") or "").split()) <= 2:
                r = min(r, 0.30)
            return r

        for o in oraciones:
            riesgos_or.append(riesgo_oracion(o))
        for fr in fragmentos:
            rfr = riesgo_fragmento(fr)
            # tope adaptativo: facil no condena solo, dificil si
            riesgos_fr.append(min(rfr, tope_fr))

        if not riesgos_or and not riesgos_fr:
            base = 0.0
        else:
            max_or = max(riesgos_or + [0.0])
            max_fr = max(riesgos_fr + [0.0])
            base = max(max_or, max_fr)
            # 1) doble evidencia: un fragmento solo no condena. Si ninguna
            # oracion llega a 0.40 y el LLM no dice mal, se topa a 0.44 (fiel).
            llm_malo = any(
                ((o.get("llm") or {}).get("veredicto") in ("contradicha", "no_respaldada"))
                for o in oraciones
            )
            if max_or < 0.40 and not llm_malo:
                base = min(base, 0.44)

        # e) 2a evidencia en dos ventanas: fuerte (0.20-0.65, alta precision:
        # similitud, atribucion, entidades, antonimos) y suave (0.35-0.65:
        # causal, universal, cifra). Los FN bajos (0.20-0.30) ya son visibles.
        extra = 0.0
        if 0.20 <= base <= 0.65:
            # similitud alta + respaldo bajo = distorsion (comparte palabras
            # pero no verdad); es lo que el conteo lexico da por bueno.
            try:
                for o in oraciones:
                    sm = max(list(o.get("similitud_por_fuente") or [0.0]) + [0.0])
                    por_f = o.get("respaldo_por_fuente") or []
                    rm = max(list(por_f) + [0.0] + [o.get("respaldo_todas", 0.0) or 0.0])
                    if sm > 0.75 and rm < 0.35:
                        extra = max(extra, 0.12)
                        break
            except Exception:
                pass
            # atribucion: el LLM cita fuente k pero el NLI solo respalda en
            # j!=k -> numero de otro (A/B, tarifas, dosis).
            try:
                for o in oraciones:
                    llm = o.get("llm") or {}
                    k = llm.get("fuente")
                    if k is None or llm.get("veredicto") != "respaldada":
                        continue
                    por_f = list(o.get("respaldo_por_fuente") or [])
                    if not por_f or k < 0 or k >= len(por_f):
                        continue
                    j = max(range(len(por_f)), key=lambda i: por_f[i])
                    if j != k and por_f[k] < 0.30 and por_f[j] > 0.55:
                        extra = max(extra, 0.12)
                        break
            except Exception:
                pass
            # 3) entidades: la cifra esta en la fuente j pero los nombres no
            # coinciden con la oracion -> dato de otro.
            try:
                for o in oraciones:
                    otxt = o.get("texto") or ""
                    if not _tokens_num(otxt):
                        continue
                    por_f = list(o.get("respaldo_por_fuente") or [])
                    if not por_f:
                        continue
                    j = max(range(len(por_f)), key=lambda i: por_f[i])
                    if por_f[j] < 0.55 or j >= len(fuentes):
                        continue
                    ent_o = set(re.findall(r"[A-ZÁÉÍÓÚ][a-záéíóú]+", otxt))
                    ent_f = set(re.findall(r"[A-ZÁÉÍÓÚ][a-záéíóú]+", fuentes[j]))
                    if ent_o and ent_f and len(ent_o & ent_f) * 2 < len(ent_o):
                        extra = max(extra, 0.12)
                        break
            except Exception:
                pass
            # 4) antonimos: contradiccion sin "no" (aumento/reduccion...).
            try:
                low_r = (respuesta or "").lower()
                low_f = (texto_fuentes or "").lower()
                for a, b in _ANTONIMOS:
                    if a in low_r and b in low_f:
                        extra = max(extra, 0.12)
                        break
            except Exception:
                pass
        if 0.35 <= base <= 0.65:
            if _CAUSAL.search(respuesta):
                extra = max(extra, 0.10)
            # universal solo si no esta en fuentes (si esta, es fiel)
            if _UNIVERSAL.search(respuesta) and not _UNIVERSAL.search(texto_fuentes):
                extra = max(extra, 0.10)
            # cifra nueva (digitos o palabra) + poco respaldo -> distorsion.
            # Si hay numero complejo en palabra ("mil novecientos..."), se
            # abstiene: el dict parcial daria FP en la parafrasis fiel.
            if not _COMPLEJO_NUM.search(respuesta):
                toks_f = _tokens_num(texto_fuentes)
                toks_r = _tokens_num(respuesta)
                if any(t not in toks_f for t in toks_r):
                    extra = max(extra, 0.08)

        # aritmetica fiel: si la cifra es suma/doble/mitad exacta de cifras de
        # fuentes, el NLI suele dar bajo y es FP -> se descuenta. Solo digitos
        # puros (las palabras van por _tokens_num y los complejos se saltan).
        try:
            nums_f = [_norm_num(n) for n in _numeros(texto_fuentes)]
            vals_f = sorted({int(x) for x in nums_f if x.isdigit() and len(x) <= 6})
            for n in _numeros(respuesta):
                nn = _norm_num(n)
                if not nn or not nn.isdigit() or len(nn) > 6 or nn in nums_f:
                    continue
                v = int(nn)
                ok = False
                s = set(vals_f)
                for a in vals_f[:40]:
                    if (v - a) in s or v == 2 * a or (a == 2 * v):
                        ok = True
                        break
                if ok:
                    extra = min(extra, 0.0) - 0.0  # marca: no es cifra inventada
                    base = max(0.0, base - 0.15)
                    break
        except Exception:
            pass

        bruto = min(1.0, max(0.0, base + extra))

        # f) ajuste adaptativo ya calculado arriba (facil -0.12, medio 0.0,
        # dificil +0.08). Acerca cada tramo a su mejor corte (0.70 / 0.50 /
        # 0.25) manteniendo el corte global 0.5.
        calibrado = bruto * 1.0 + ajuste
        if bruto > 0.75:
            calibrado = max(calibrado, 0.85)
        return float(min(1.0, max(0.0, calibrado)))
