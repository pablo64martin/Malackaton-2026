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

_CAUSAL = re.compile(
    r"gracias a|debido a|a causa de|porque|por ello|esto permiti|esto provoc|"
    r"esto gener|esto llev|esto hizo posible",
    re.IGNORECASE,
)
_UNIVERSAL = re.compile(
    r"\btodos\b|\btodas\b|\btodo el\b|\bsiempre\b|\bnunca\b|\bning[uú]n",
    re.IGNORECASE,
)


def _numeros(s):
    return re.findall(r"\d+(?:[.,]\d+)?", s or "")


def _norm_num(t):
    return (t or "").replace(".", "").replace(",", "").replace(" ", "")


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

        def riesgo_oracion(u):
            por_f = u.get("respaldo_por_fuente") or []
            mejor = max(list(por_f) + [0.0])
            todas = u.get("respaldo_todas", 0.0) or 0.0
            # a) no castigar combinacion de fuentes
            respaldo = mejor if mejor > todas else todas

            # d) LLM como puerta: en H el NLI solo da alto a casi todo
            # (fiel 0.39 vs alucina 0.30), el LLM-mal tiene precision ~95%.
            llm = u.get("llm") or {}
            v = llm.get("veredicto")
            conf = llm.get("confianza", 0.0) or 0.0
            if v in ("contradicha", "no_respaldada"):
                r = 0.80 + 0.15 * conf if conf >= 0.5 else 0.60
            elif v == "respaldada":
                if respaldo >= 0.40:
                    # NLI acompana: probablemente fiel -> riesgo bajo
                    r = 0.15 + (1.0 - respaldo) * 0.4
                elif respaldo >= 0.20:
                    # zona gris H: aqui caen muchas alucinaciones sutiles con
                    # LLM perdonando. Como FN pesa 4x, se pone por encima
                    # del corte (0.52) en vez de por debajo.
                    r = 0.52
                else:
                    r = 0.65  # NLI muy bajo: desconfiar del "respaldada"
            else:
                r = (1.0 - respaldo) * 0.7

            # c) contradiccion (en H casi no separa: solo picos fuertes)
            por_c = u.get("contradiccion_por_fuente") or []
            max_c = max(list(por_c) + [0.0])
            todas_c = u.get("contradiccion_todas", 0.0) or 0.0
            if todas_c > 0.65 or max_c > 0.75:
                r = max(r, 0.85)
            return r

        def riesgo_fragmento(u):
            # b) fragmentos amortiguados: su peor respaldo_todas mediano en
            # fieles H es 0.17 -> sin amortiguar darian 0.83 a la mitad de
            # fieles. Se escala a la mitad y con tope.
            por_f = u.get("respaldo_por_fuente") or []
            mejor = max(list(por_f) + [0.0])
            todas = u.get("respaldo_todas", 0.0) or 0.0
            respaldo = mejor if mejor > todas else todas
            r = (1.0 - respaldo) * 0.55
            por_c = u.get("contradiccion_por_fuente") or []
            max_c = max(list(por_c) + [0.0])
            todas_c = u.get("contradiccion_todas", 0.0) or 0.0
            if todas_c > 0.7 or max_c > 0.8:
                r = max(r, 0.60)
            # ignorar micro-fragmentos de 1-2 palabras (ruido NLI)
            if len((u.get("texto") or "").split()) <= 2:
                r = min(r, 0.40)
            return r

        for o in oraciones:
            riesgos_or.append(riesgo_oracion(o))
        for fr in fragmentos:
            riesgos_fr.append(riesgo_fragmento(fr))

        if not riesgos_or and not riesgos_fr:
            base = 0.0
        else:
            # b) el peor manda, pero la oracion pesa mas que el fragmento
            base = max(riesgos_or + riesgos_fr + [0.0])

        # e) heuristicas: solo empujan si ya hay sospecha (base media)
        extra = 0.0
        if 0.25 <= base < 0.75:
            if _CAUSAL.search(respuesta):
                extra = max(extra, 0.10)
            if _UNIVERSAL.search(respuesta):
                extra = max(extra, 0.08)
            # cifra nueva + poco respaldo -> posible distorsion
            nuevos = False
            src_norm = _norm_num(texto_fuentes)
            for n in _numeros(respuesta):
                nn = _norm_num(n)
                if nn and nn not in src_norm:
                    nuevos = True
                    break
            if nuevos and base >= 0.35:
                extra = max(extra, 0.08)

        bruto = min(1.0, base + extra)

        # f) calibracion: los crudos salen bajos (~0.1-0.3) y el corte es 0.5.
        # FN pesa 4x -> ser agresivo.
        calibrado = bruto * 1.0 + 0.05
        if bruto > 0.75:
            calibrado = max(calibrado, 0.85)
        return float(min(1.0, max(0.0, calibrado)))
