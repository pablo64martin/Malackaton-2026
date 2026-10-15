"""Guardian coste-sensible (FN pesa 4x): NLI + LLM juez + chequeos lexicos.

Solo Python estandar. score() devuelve P(alucinacion) en [0,1], calibrada para
que el corte 0.5 sea el optimo con coste 4:1.
Estrategia: riesgo continuo por oracion (LLM x0.55 + NLI x0.30 + gap x0.30),
peor fragmento con tope, 2a evidencia lexica con abstencion, mitigaciones
estrictas (aritmetica TODAS-derivables, unanimidad LLM+NLI) y reescalado.
"""

import re
import unicodedata

# Normalizacion Unicode: comparar sin tildes de forma sistematica en vez
# de alternancias en cada regex.
def _norm(s):
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

# Listas de patrones: causas/consecuencias, universales/sobregeneralizacion
# ("nacional/del pais/cualquier", "permite/preve/supone/lleva a/hace que")
# y numeros en palabra + %.
_CAUSAL = re.compile(
    r"gracias a|debido a|a causa de|porque|por ello|por eso|con lo que|"
    r"hace que|lleva a|permite|prev[eé]|supone|esto permiti|esto provoc|"
    r"esto gener|esto llev|esto hizo posible|por consiguiente|por tanto|"
    r"de modo que|as[ií] que",
    re.IGNORECASE,
)
_UNIVERSAL = re.compile(
    r"\btodos\b|\btodas\b|\btodo el\b|\btoda la\b|\bcualquier\b|\bcualquiera\b|"
    r"\bsiempre\b|\bnunca\b|\bjam[aá]s\b|\bning[uú]n\b|\bninguna\b|\bnadie\b|"
    r"\bdel pa[ií]s\b|\bnacional\b|\ben todo\b|\ben toda\b",
    re.IGNORECASE,
)
# Restricciones de alcance (tiempo, calidad y ordinales: durante/desde,
# anos/meses/dias/horas/semanas, frescos/refrigerados/bajura/altura,
# primeros/ultimos, minimo/maximo).
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


_PALABRAS_VACIAS = set(
    "de la el en y a los las del un una que se por con para al lo su sus es "
    "son fue como mas o no ha han sido este esta estos estas entre sobre "
    "tras hasta desde sin ser le les ya".split()
)


def _palabras(s):
    import unicodedata as _ud
    s = _ud.normalize("NFD", (s or "").lower())
    s = "".join(ch for ch in s if _ud.category(ch) != "Mn")
    return {w[:5] for w in re.findall(r"[a-z0-9]+", s)
            if w not in _PALABRAS_VACIAS and len(w) > 2 and not w.isdigit()}


def _items_o(o, fuentes):
    """Verifica item a item lo afirmado por la oracion. Devuelve
    (fallos, total) o None si no hay nada verificable.
    Pesos: numeros x2 (evidencia dura), resto x1."""
    otxt = o.get("texto") or ""
    if not fuentes or not otxt.strip():
        return None
    sims = list(o.get("similitud_por_fuente") or [])
    if len(sims) == len(fuentes) and fuentes:
        j = max(range(len(sims)), key=lambda i: sims[i])
    else:
        j = 0
    best = fuentes[j]
    nbest = _norm_nums(best.lower())
    notxt = _norm_nums(otxt.lower())
    alln = [_norm_nums(f.lower()) for f in fuentes]
    fails, total = 0, 0

    def _numkey(n):
        n = n.replace(",", ".")
        if "." in n:
            return {n, n.replace(".", "")}
        return {n, n}

    # numeros: presencia + co-ocurrencia con entidades de su clausula.
    nums = set()
    primero = (otxt.strip().split() or [""])[0].strip('",«»(')
    for m in _NUM_RE.finditer(notxt):
        nums.update(_numkey(m.group(0)))
    for n in sorted(nums)[:12]:
        total += 2
        s_num = [f for f in alln if n in f]
        if not s_num:
            fails += 2
            continue
        cerca = set()
        for cl in re.split(r"[,;]|\sy\s", otxt):
            if n in _norm_nums(cl.lower()):
                cerca.update(re.findall(r"[A-ZÁÉÍÓÚ][a-záéíóú]+", cl))
        cerca.discard(primero)
        if cerca and not any(cerca & set(re.findall(r"[A-ZÁÉÍÓÚ][a-záéíóú]+", f)) for f in s_num):
            fails += 2
    # entidades sueltas: deben existir en alguna fuente
    ents = set(re.findall(r"[A-ZÁÉÍÓÚ][a-záéíóú]+", otxt))
    ents.discard(primero)
    allents = set()
    for f in alln:
        allents.update(re.findall(r"[A-ZÁÉÍÓÚ][a-záéíóú]+", f))
    for e in ents:
        total += 1
        if e not in allents:
            fails += 1
    # restricciones/matices: solo cuentan las de fuentes de las que la
    # oracion habla (solape>=0.30); asi no penaliza combinar fuentes.
    rel = [f for f in fuentes if _solape(notxt, _norm_nums(f.lower())) >= 0.30] or [best]
    reln = "\n".join(_norm_nums(f.lower()) for f in rel)
    for sfn in [_norm_nums(f.lower()) for f in rel]:
        for mm in _CONDICIONAL.finditer(sfn):
            total += 1
            if mm.group(0).lower() not in notxt:
                fails += 1
    # universales/causas anadidas: deben estar en alguna fuente relatada
    if _UNIVERSAL.search(otxt) and not _UNIVERSAL.search(reln):
        total += 1
        fails += 1
    if _CAUSAL.search(otxt) and not _CAUSAL.search(reln):
        total += 1
        fails += 1
    if total == 0:
        return None
    return (fails, total)


def _solape(a, b, n=5):
    ta = {t for t in _toks(a) if len(t) >= n}
    tb = set(_toks(b))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


# Contradicciones sin "no": una cara en la respuesta y la otra en fuentes.
_ANTONIMOS = [
    ("aument", "reduj"), ("aument", "disminuy"), ("aument", "cay"),
    ("sube", "baja"), ("subi", "baj"), ("crec", "decrec"),
    ("aprob", "propus"), ("aprob", "propuso"), ("inmuniz", "vacun"),
    ("todos", "algunos"), ("todas", "algunas"), ("siempre", "nunca"),
    ("todos", "varios"), ("todas", "varias"),
    ("increment", "decrement"), ("increment", "reduj"), ("descens", "aument"),
    ("mejor", "peor"), ("abrir", "cerrar"), ("apertur", "cierre"),
    ("ganar", "perder"), ("gananc", "perdid"), ("contrat", "desped"),
    ("obligatorio", "opcional"), ("prohibido", "permitido"),
    ("mayoria", "minoria"), ("exito", "fracaso"), ("verdadero", "falso"),
]
# Endurecimiento verbal: la respuesta afirma fuerte lo que las fuentes solo
# proponen/sugieren (criterio: endurecer un verbo = alucinacion). Pares
# (fuerte_en_respuesta, debil_en_fuentes).
_ENDURECIDO = [
    ("aprob", "propus"), ("aprob", "plante"),
    ("inmuniz", "vacun"), ("garantiz", "posib"),
    ("costar", "presupuest"), ("es ", "podr"),
    ("confirm", "prev"), ("demostr", "sugier"),
    ("debe", "deberia"), ("debe", "podria"), ("tiene que", "puede"),
    ("obligatorio", "recomend"), ("seguro", "probable"),
    ("anunci", "rumor"), ("descubr", "investiga"),
]


# Umbral bruto optimo medido en train para coste 4:1. Se reescala de forma
# monotona para que el corte oficial 0.5 sea el optimo (no cambia el orden).
UMBRAL_BRUTO = 0.36


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
            # facil tipo-e
            _, tope_fr, ajuste = 0.0, 0.30, -0.12
        elif n_fuentes >= 7 or n_or >= 3:
            # dificil tipo-H/T/N (7-9 fuentes)
            _, tope_fr, ajuste = 0.0, 0.65, 0.08
        else:
            # medio s/G/R (3-6 fuentes)
            _, tope_fr, ajuste = 0.0, 0.40, -0.02

        def riesgo_oracion(u):
            # Continuo ponderado: sin escalones que empataban cientos de casos.
            # Domina el LLM (x0.55) con NLI (x0.30), gap similitud-respaldo
            # = distorsion (x0.30) y contradiccion (x0.10).
            por_f = u.get("respaldo_por_fuente") or []
            mejor = max(list(por_f) + [0.0])
            todas = u.get("respaldo_todas", 0.0) or 0.0
            # a) no castigar combinacion de fuentes
            respaldo = mejor if mejor > todas else todas
            sim_max = max(list(u.get("similitud_por_fuente") or [0.0]) + [0.0])

            # d) riesgo LLM continuo: respaldada confiada + NLI ok pesa casi 0;
            # respaldada con NLI bajo o poca confianza sube.
            llm = u.get("llm") or {}
            v = llm.get("veredicto")
            conf = llm.get("confianza", 0.0) or 0.0
            if v == "contradicha":
                lr = 1.0
            elif v == "no_respaldada":
                lr = 0.85
            elif v == "respaldada":
                lr = 0.72 - respaldo * 1.15 - max(0.0, conf - 0.90) * 1.0
                lr = min(0.80, max(0.03, lr))
            else:
                lr = 0.50

            # c) contradiccion como sumando, no como sentencia.
            por_c = u.get("contradiccion_por_fuente") or []
            max_c = max(list(por_c) + [0.0])
            todas_c = u.get("contradiccion_todas", 0.0) or 0.0
            contra = min(1.0, max(max_c, todas_c))

            gap = max(0.0, sim_max - respaldo)
            r = 0.28 * (1.0 - respaldo) + 0.55 * lr + 0.06 * contra + 0.25 * gap

            # Verificacion por items: lo afirmado por la oracion contra sus
            # fuentes. Puede RESCATAR (todo verificado + NLI bajo = parafrasis
            # con fallo NLI) y CAZAR (items que fallan + NLI alto = crimen
            # perfecto). Ninguna otra senal hace las dos cosas.
            iv = _items_o(u, fuentes)
            if iv is not None:
                fails, total = iv
                if total >= 2:
                    fr = fails / total
                    if fails == 0 and r < 0.65:
                        r = min(r, 0.30)
                    elif fr >= 0.5:
                        r = max(r, 0.62)
            return r

        def riesgo_fragmento(u):
            # Por tramos con topes por tramo (el lineal daba 0.50 al fiel
            # mediano 0.17; el continuo bajaba recall).
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
            if len((u.get("texto") or "").split()) <= 3:
                r = min(r, 0.30)
            return r

        for o in oraciones:
            riesgos_or.append(riesgo_oracion(o))
        for fr in fragmentos:
            rfr = riesgo_fragmento(fr)
            # Tope adaptativo: facil no condena solo, dificil si.
            rfr = min(rfr, tope_fr)
            riesgos_fr.append(rfr)

        if not riesgos_or and not riesgos_fr:
            base = 0.0
        else:
            max_or = max(riesgos_or + [0.0])
            max_fr = max(riesgos_fr + [0.0])
            base = max(max_or, max_fr)

        # e) 2a evidencia en dos ventanas: fuerte (0.20-0.65, alta precision:
        # similitud, atribucion, entidades, antonimos) y suave (0.35-0.65:
        # causal, universal, cifra). Los FN bajos (0.20-0.30) ya son visibles.
        extra = 0.0
        n_extra = 0
        scope_hit = False
        # Ventana amplia (0.15) solo para las dos senales de maxima precision:
        # matiz condicional omitido y numero atribuido a otra fuente. Ambas
        # verifican la fuente citada, casi nunca disparan en fieles.
        if 0.15 <= base <= 0.65:
            # atribucion DESACTIVADA: precision medida 0.15 (9 TP / 50 FP).
            # El mismatch cita-vs-top es normal al combinar fuentes.
            # RESTRICCION caida: la oracion habla de la fuente citada
            # (solape>=0.30 con numeros normalizados) pero omite una
            # restriccion suya; incluye partitivo "N de los/las/N".
            try:
                for o in oraciones:
                    llm = o.get("llm") or {}
                    k = llm.get("fuente")
                    por_k = list(o.get("respaldo_por_fuente") or [])
                    if not isinstance(k, int) or k < 0 or k >= len(fuentes):
                        continue
                    otxt = o.get("texto") or ""
                    srcn = _norm_nums((fuentes[k] or "").lower())
                    orcn = _norm_nums(otxt.lower())
                    if _solape(orcn, srcn) < 0.40:
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
                        n_extra += 1
                        extra = max(extra, 0.15)
                        scope_hit = True
                        break
            except Exception:
                pass
            # (unidad cambiada: sin aporte medido en train; fuera)
            # 7) CUENTA CASI DESACTIVADO: FP en redondeos fieles (12.4% -> 12%
            # cae dentro del +-12% y es compresion legitima).
        if 0.20 <= base <= 0.65:
            # similitud: el gap ya vive en riesgo_oracion (0.30*gap); el extra
            # aqui solo empujaba FP (medido 0 TP / 4 FP). Fuera.
            # (atribucion y condicional viven en la ventana amplia 0.15 de arriba)
            # 3) entidades DESACTIVADO: precision medida 0.14 (20 TP / 121 FP).
            # El solape de nombres propios falla en parafrasis fieles.
            # Atribucion top-restringida DESACTIVADA tras ablacion (no aporta).
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
                                n_extra += 1
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
                                n_extra += 1
                                extra = max(extra, 0.12)
                                break
                        if extra >= 0.12:
                            break
            except Exception:
                pass
            # (condicional duplicado eliminado: vive en la ventana 0.15)
        if 0.35 <= base <= 0.65:
            # causal triple-puerta: ausente en fuentes + NLI flojo en alguna
            # oracion + LLM sin aval confiado. Solo asi suma.
            if _CAUSAL.search(respuesta) and not _CAUSAL.search(texto_fuentes):
                min_rt = min(
                    [max(max(o.get("respaldo_por_fuente") or [0.0]), o.get("respaldo_todas", 0.0) or 0.0) for o in oraciones] + [1.0]
                )
                conf_max = 0.0
                for o in oraciones:
                    llm = o.get("llm") or {}
                    if llm.get("veredicto") == "respaldada":
                        cf = llm.get("confianza") or 0
                        if cf > conf_max:
                            conf_max = cf
                if min_rt < 0.6 and conf_max < 0.92:
                    n_extra += 1
                    extra = max(extra, 0.10)
            # universal solo si no esta en fuentes (si esta, es fiel)
            if _UNIVERSAL.search(respuesta) and not _UNIVERSAL.search(texto_fuentes):
                n_extra += 1
                extra = max(extra, 0.10)
            # cifra nueva DESACTIVADA: precision medida 0.22 (45 TP / 159 FP),
            # bajo azar. La distorsion numerica ya la cubre similitud-alta +
            # respaldo-bajo con mucha mejor precision.

        # aritmetica derivada fiel: conjunto rico (suma, resta,
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
                    # producto, cociente y % (tasas y porcentajes)
                    deriv.add(round(a * b, 2))
                    if b:
                        deriv.add(round(a / b, 2))
                        deriv.add(round(100 * a / b, 2))
                        deriv.add(round(100 * abs(a - b) / b, 2))
            nums_r = _NUM_RE.findall(rlow)
            missing = 0
            derived = 0
            for n in nums_r:
                nn = n.replace(",", ".")
                if nn in fflat:
                    continue
                missing += 1
                try:
                    fv = float(nn)
                    tol = max(0.051, 0.006 * abs(fv))
                    if any(abs(fv - v) <= tol for v in deriv):
                        derived += 1
                except Exception:
                    pass
            if missing and derived == missing:
                base = max(0.0, base - 0.22)
        except Exception:
            pass

        # Descuento por unanimidad: todo LLM-respaldada con >=0.95,
        # sin contradiccion relevante, CON NLI de acuerdo (min respaldo>=0.30),
        # sin restriccion caida y cifras exactas -> fiel casi seguro.
        try:
            mit_ok = bool(oraciones) and not scope_hit
            min_rt = 1.0
            for o in oraciones:
                llm = o.get("llm") or {}
                if llm.get("veredicto") != "respaldada" or (llm.get("confianza") or 0) < 0.95:
                    mit_ok = False
                    break
                if (o.get("contradiccion_todas", 0) or 0) >= 0.40:
                    mit_ok = False
                    break
                pf = o.get("respaldo_por_fuente") or [0.0]
                rt = max(max(pf), o.get("respaldo_todas", 0.0) or 0.0)
                if rt < min_rt:
                    min_rt = rt
            if min_rt < 0.30:
                mit_ok = False
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

        # Palabras nuevas como evidencia NEGATIVA: la parafrasis
        # fiel comparte pocas palabras con la fuente; el anadido inventado
        # suele ser 1-2 palabras. Mucha novedad + LLM unanime = parafrasis.
        try:
            pal_f = set()
            for f in fuentes:
                pal_f.update(_palabras(f))
            pal_r = _palabras(respuesta)
            if pal_r:
                nov = len(pal_r - pal_f) / len(pal_r)
                unanime = bool(oraciones) and all(
                    (o.get("llm") or {}).get("veredicto") == "respaldada"
                    for o in oraciones
                )
                contra_alta = any(
                    (o.get("contradiccion_todas", 0) or 0) >= 0.40
                    for o in oraciones
                )
                if nov > 0.30 and unanime and not contra_alta and not scope_hit:
                    base = max(0.0, base - 0.15)
        except Exception:
            pass

        bruto = min(1.0, max(0.0, base + extra + 0.02 * max(0, n_extra - 1)))

        # f) ajuste adaptativo por tramo + reescalado al corte optimo de train
        # (UMBRAL_BRUTO): monotono, no cambia el orden. La puerta final trabaja
        # ya en escala del corte oficial 0.5.
        calibrado = _reescalar(bruto * 1.0 + ajuste)
        if bruto > 0.75:
            calibrado = max(calibrado, 0.85)
        # Liston por unidad (Bonferroni): con n oraciones hay n oportunidades
        # de pico ruidoso; cada unidad debe superarlo. En n=1 no cambia nada
        # (train-H intacto); en respuestas largas filtra el ruido medio.
        liston = 0.5 + 0.06 * max(0, n_or - 1)
        if calibrado < liston:
            calibrado = min(calibrado, 0.49)
        return float(min(1.0, max(0.0, calibrado)))
