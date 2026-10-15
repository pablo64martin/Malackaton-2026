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
    r"\bsiempre\b|\bnunca\b|\bjam[aá]s\b|\bning[uú]n\b|\bninguna\b|\bnadie\b|"
    r"\bdel pa[ií]s\b|\bnacional\b|\ben todo\b|\ben toda\b",
    re.IGNORECASE,
)
# De Marcos: restricciones de alcance (las suyas traen tiempo, calidad y
# ordinales que faltaban: durante/desde, anos/meses/dias/horas/semanas,
# frescos/refrigerados/bajura/altura, primeros/ultimos, minimo/maximo).
# Se compara fuente CITADA vs oracion con numeros normalizados y puerta de
# solape: solo cuenta si la oracion habla de esa fuente.
_CONDICIONAL = re.compile(
    r"podr[ií]a|podr[ií]an|puede(n)?|posible(mente)?|probable(mente)?|"
    r"si se cumple|si llueve|siempre que|salvo|excepto|"
    r"s[oó]lo|[uú]nicamente|al menos|en caso de|si se aprueba|"
    r"condici[oó]n|condicionado|m[aá]s de|menos de|alrededor de|"
    r"aproximadamente|hasta|entre|durante|desde|"
    r"frescos?|refrigerad\w*|bajura|altura|primeros?|[uú]ltimos?|"
    r"m[íi]nimo|m[aá]ximo|a[ñn]os?|meses|d[íi]as|horas|semanas?",
    re.IGNORECASE,
)
_DE_RE = re.compile(r"(\d+)\s+de\s+(los|las|\d+)", re.IGNORECASE)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*%?")
_UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(millones?|miles?|millar\w*|p[aá]ginas?|documentos?|"
    r"toneladas?|kilos?|euros?|c[eé]ntimos?|metros?|km|tiendas?|arboles?|árboles?|"
    r"ejemplares?|a[ñn]os?|meses|d[íi]as?)",
    re.IGNORECASE,
)
_PALABRA_NUM = {
    "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "dieciséis": 16, "diecisiete": 17, "dieciocho": 18,
    "diecinueve": 19, "veinte": 20, "treinta": 30,
    "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
    "ochenta": 80, "noventa": 90, "cien": 100, "ciento": 100,
    "mil": 1000, "una": 1,
}


def _norm_nums(s):
    """Minusculas con numeros en palabra -> digitos (para comparar alcance)."""
    out = s or ""
    for w, v in _PALABRA_NUM.items():
        out = re.sub(r"\b" + w + r"\b", str(v), out)
    return out


def _toks(s):
    return re.findall(r"[a-záéíóúñü0-9]+", (s or "").lower())


def _solape(a, b, n=5):
    ta = {t for t in _toks(a) if len(t) >= n}
    tb = set(_toks(b))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)
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
# Endurecimiento verbal: la respuesta afirma fuerte lo que las fuentes solo
# proponen/sugieren (criterio: endurecer un verbo = alucinacion). Pares
# (fuerte_en_respuesta, debil_en_fuentes).
_ENDURECIDO = [
    ("aprob", "propus"), ("aprob", "plante"),
    ("inmuniz", "vacun"), ("garantiz", "posib"),
    ("costar", "presupuest"), ("es ", "podr"),
    ("confirm", "prev"), ("demostr", "sugier"),
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


# Umbral bruto optimo medido en train para coste 4:1. Se reescala de forma
# monotona para que el corte oficial 0.5 sea el optimo (no cambia el orden).
UMBRAL_BRUTO = 0.33


def _reescalar(s):
    t = UMBRAL_BRUTO
    s = min(1.2, max(0.0, float(s)))
    if s < t:
        return (s / t) * 0.5 if t > 0 else 0.0
    return 0.5 + ((s - t) / (1.2 - t)) * 0.5


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
            gris, tope_fr, ajuste = 0.46, 0.40, -0.02

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
                    r = gris  # adaptativa: 0.38 facil / 0.46 medio / 0.58 dificil
                else:
                    # NLI muy bajo: en dificil cruza solo (recall x4); en
                    # facil/medio necesita 2a evidencia (ahi FN~0 y el NLI
                    # falla en parafrasis fieles).
                    r = 0.65 if nivel == 2 else 0.52
            else:
                r = (1.0 - respaldo) * 0.7

            # c) contradiccion en picos fuertes pero SIN condenar sola: el
            # NLI-contradiccion tambien falla en fieles. Deja 0.60 y cuenta
            # como evidencia en la puerta final (contra_pico).
            por_c = u.get("contradiccion_por_fuente") or []
            max_c = max(list(por_c) + [0.0])
            todas_c = u.get("contradiccion_todas", 0.0) or 0.0
            if (todas_c > 0.65 or max_c > 0.75) and r > 0.35:
                r = max(r, 0.60)
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
        # Señal de maxima precision: el LLM dice mal (solo ~20 FP en train).
        # Ella sola puede condenar; el resto necesita companera (puerta final).
        llm_malo = any(
            ((o.get("llm") or {}).get("veredicto") in ("contradicha", "no_respaldada"))
            for o in oraciones
        )
        # Pico de contradiccion NLI (para la puerta de doble evidencia).
        contra_pico = False
        for u in list(oraciones) + list(fragmentos):
            por_c = u.get("contradiccion_por_fuente") or []
            mc = max(list(por_c) + [0.0])
            tc = u.get("contradiccion_todas", 0.0) or 0.0
            if tc > 0.65 or mc > 0.75:
                contra_pico = True
                break

        # e) 2a evidencia en dos ventanas: fuerte (0.20-0.65, alta precision:
        # similitud, atribucion, entidades, antonimos) y suave (0.35-0.65:
        # causal, universal, cifra). Los FN bajos (0.20-0.30) ya son visibles.
        extra = 0.0
        scope_hit = False
        # Ventana amplia (0.15) solo para las dos senales de maxima precision:
        # matiz condicional omitido y numero atribuido a otra fuente. Ambas
        # verifican la fuente citada, casi nunca disparan en fieles.
        if 0.15 <= base <= 0.65:
            # atribucion DESACTIVADA: precision medida 0.15 (9 TP / 50 FP).
            # El mismatch cita-vs-top es normal al combinar fuentes.
            # RESTRICCION caida (Marcos): la oracion habla de la fuente citada
            # (solape>=0.30 con numeros normalizados) pero omite una
            # restriccion suya; incluye partitivo "N de los/las/N".
            try:
                for o in oraciones:
                    llm = o.get("llm") or {}
                    k = llm.get("fuente")
                    if not isinstance(k, int) or k < 0 or k >= len(fuentes):
                        continue
                    otxt = o.get("texto") or ""
                    srcn = _norm_nums((fuentes[k] or "").lower())
                    orcn = _norm_nums(otxt.lower())
                    if _solape(orcn, srcn) < 0.30:
                        continue
                    miss = False
                    for mm in _CONDICIONAL.finditer(srcn):
                        if mm.group(0).lower() not in orcn:
                            miss = True
                            break
                    if not miss and _DE_RE.search(srcn):
                        ns = _NUM_RE.findall(srcn)
                        nr = _NUM_RE.findall(orcn)
                        if ns and not any(n in nr for n in ns):
                            miss = True
                    if miss:
                        extra = max(extra, 0.15)
                        scope_hit = True
                        break
            except Exception:
                pass
            # (unidad cambiada: sin aporte medido en train; fuera)
            # 7) CUENTA CASI DESACTIVADO: FP en redondeos fieles (12.4% -> 12%
            # cae dentro del +-12% y es compresion legitima).
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
            # (atribucion y condicional viven en la ventana amplia 0.15 de arriba)
            # 3) entidades DESACTIVADO: precision medida 0.14 (20 TP / 121 FP).
            # El solape de nombres propios falla en parafrasis fieles.
            # 4) antonimos por oracion y su fuente top: contradiccion sin "no".
            # Global daba FP en textos contrastivos fieles (fuentes que hablan
            # de aumento Y reduccion para entidades distintas). Solo cuenta en
            # base media-alta para no inventar desde fieles claros.
            try:
                if base >= 0.30:
                    for o in oraciones:
                        otxt = (o.get("texto") or "").lower()
                        por_f = list(o.get("respaldo_por_fuente") or [])
                        j = max(range(len(por_f)), key=lambda i: por_f[i]) if por_f else -1
                        fj = (fuentes[j] or "").lower() if 0 <= j < len(fuentes) else ""
                        for a, b in _ANTONIMOS:
                            if a in otxt and b in fj:
                                extra = max(extra, 0.12)
                                break
                        if extra >= 0.12:
                            break
            except Exception:
                pass
            # 4b) endurecimiento verbal: la oracion afirma fuerte lo que su
            # fuente top solo propone/sugiere (criterio escrito).
            try:
                if base >= 0.20:
                    for o in oraciones:
                        otxt = (o.get("texto") or "").lower()
                        por_f = list(o.get("respaldo_por_fuente") or [])
                        j = max(range(len(por_f)), key=lambda i: por_f[i]) if por_f else -1
                        fj = (fuentes[j] or "").lower() if 0 <= j < len(fuentes) else ""
                        for fuerte, debil in _ENDURECIDO:
                            if fuerte in otxt and debil in fj and fuerte not in fj:
                                extra = max(extra, 0.12)
                                break
                        if extra >= 0.12:
                            break
            except Exception:
                pass
            # (condicional duplicado eliminado: vive en la ventana 0.15)
        if 0.35 <= base <= 0.65:
            # causal triple-puerta (Marcos): ausente en fuentes + NLI flojo en
            # alguna oracion + LLM sin aval confiado. Solo asi suma.
            if _CAUSAL.search(respuesta) and not _CAUSAL.search(texto_fuentes):
                min_rt = min(
                    [o.get("respaldo_todas", 0.0) or 0.0 for o in oraciones] + [1.0]
                )
                conf_max = 0.0
                for o in oraciones:
                    llm = o.get("llm") or {}
                    if llm.get("veredicto") == "respaldada":
                        cf = llm.get("confianza") or 0
                        if cf > conf_max:
                            conf_max = cf
                if min_rt < 0.5 and conf_max < 0.92:
                    extra = max(extra, 0.10)
            # universal solo si no esta en fuentes (si esta, es fiel)
            if _UNIVERSAL.search(respuesta) and not _UNIVERSAL.search(texto_fuentes):
                extra = max(extra, 0.10)
            # cifra nueva DESACTIVADA: precision medida 0.22 (45 TP / 159 FP),
            # bajo azar. La distorsion numerica ya la cubre similitud-alta +
            # respaldo-bajo con mucha mejor precision.

        # aritmetica derivada fiel (Marcos): conjunto rico (suma, resta,
        # 100-a-b para porcentajes, x100, x1000) y condicion estricta: TODAS
        # las cifras no literales deben ser derivables. Si no, no descuenta.
        # Asi no perdona alucinaciones con una coincidencia casual.
        try:
            fflat = _norm_nums(texto_fuentes.lower()).replace(",", ".")
            rlow = _norm_nums(respuesta.lower())
            src_vals = []
            for x in _NUM_RE.findall(fflat):
                try:
                    src_vals.append(float(x.replace(",", ".")))
                except Exception:
                    pass
            deriv = {round(v, 2) for v in src_vals}
            for i in range(min(len(src_vals), 12)):
                for j in range(min(len(src_vals), 12)):
                    if i == j:
                        continue
                    a, b = src_vals[i], src_vals[j]
                    deriv.add(round(a + b, 2))
                    deriv.add(round(abs(a - b), 2))
                    deriv.add(round(100 - a - b, 2))
                    deriv.add(round(a * 100, 2))
                    deriv.add(round(a * 1000, 2))
            nums_r = _NUM_RE.findall(rlow)
            missing = 0
            derived = 0
            for n in nums_r:
                nn = n.replace(",", ".")
                if nn in fflat:
                    continue
                missing += 1
                try:
                    if round(float(nn), 2) in deriv:
                        derived += 1
                except Exception:
                    pass
            if missing and derived == missing:
                base = max(0.0, base - 0.20)
        except Exception:
            pass

        # Descuento por unanimidad (Marcos): todo LLM-respaldada con >=0.95,
        # sin contradiccion relevante, sin restriccion caida y cifras exactas
        # -> fiel casi seguro. Recorta FP en claras sin tocar FN.
        try:
            mit_ok = bool(oraciones) and not scope_hit
            for o in oraciones:
                llm = o.get("llm") or {}
                if llm.get("veredicto") != "respaldada" or (llm.get("confianza") or 0) < 0.95:
                    mit_ok = False
                    break
                if (o.get("contradiccion_todas", 0) or 0) >= 0.40:
                    mit_ok = False
                    break
            if mit_ok:
                nums2 = _NUM_RE.findall(_norm_nums(respuesta.lower()))
                fflat2 = _norm_nums(texto_fuentes.lower()).replace(",", ".")
                # Sin cifras: la unanimidad confiada + sin contradiccion ya
                # basta (el texto fiel sin numeros rara vez engana al LLM en
                # TODAS las frases con >=0.95).
                if not nums2 or all(n.replace(",", ".") in fflat2 for n in nums2):
                    base = max(0.0, base - 0.20)
        except Exception:
            pass

        bruto = min(1.0, max(0.0, base + extra))

        # f) ajuste adaptativo por tramo + reescalado al corte optimo de train
        # (UMBRAL_BRUTO): monotono, no cambia el orden. La puerta final trabaja
        # ya en escala del corte oficial 0.5.
        calibrado = _reescalar(bruto * 1.0 + ajuste)
        if bruto > 0.75:
            calibrado = max(calibrado, 0.85)
        # Puerta final DESACTIVADA: las mitigaciones estrictas (aritmetica
        # TODAS-derivables, unanimidad 0.95) ya recortan FP con precision;
        # la puerta se comia recall de grises con 1 evidencia (FN x4).
        if False and not llm_malo and calibrado < 0.55:
            frag_muy_bajo = any(
                len((fr.get("texto") or "").split()) > 2
                and max(max(list(fr.get("respaldo_por_fuente") or [0.0]) + [0.0]),
                        fr.get("respaldo_todas", 0.0) or 0.0) < 0.08
                for fr in fragmentos
            )
            n_evid = (1 if max_or >= 0.50 else 0) + (1 if frag_muy_bajo else 0) \
                + (1 if extra >= 0.10 else 0) + (1 if contra_pico else 0)
            if n_evid < 2:
                calibrado = min(calibrado, 0.49)
        return float(min(1.0, max(0.0, calibrado)))
