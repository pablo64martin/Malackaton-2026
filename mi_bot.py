"""El Guardian - detector de alucinaciones (solo Python estandar).

Estrategia:
1. Peor unidad manda: basta 1 afirmacion falsa (max de riesgo, no promedio).
2. Fragmentos rescatan fieles que combinan varias fuentes.
3. Desconfia del LLM 'respaldada' con confianza baja o contradiccion NLI alta.
4. Chequeos propios de texto: numeros, alcance (todos/cualquier) y causas
   inventadas, que es donde NLI y LLM mas fallan en H.
5. Calibracion agresiva: escapar cuesta 4x, asi que se empuja todo hacia arriba
   para que el corte optimo caiga en 0.5.
"""

import re

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
_UNIVERSALES = [
    "todos", "todas", "todo el", "toda la", "cualquier", "cualquiera",
    "siempre", "nunca", "jamas", "jamás", "ningun", "ningún", "ninguna",
    "del pais", "del país", "nacional", "en todo", "en toda",
]
_CAUSALES = [
    "gracias a", "debido a", "por eso", "con lo que", "hace que",
    "lleva a", "permite", "preve", "prevé", "supone",
]


def _norm_num(s):
    s = s.strip().replace("%", "")
    s = s.replace(" ", "")
    # 28.000 -> 28000 ; 2,6 -> 2.6 ; 5,05 -> 5.05 ; 1.380 -> 1380
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        # coma decimal o miles: si 3 digitos al final, es miles
        partes = s.split(",")
        if len(partes) == 2 and len(partes[1]) == 3 and len(partes[0]) <= 3:
            s = partes[0] + partes[1]
        else:
            s = s.replace(",", ".")
    elif "." in s:
        partes = s.split(".")
        # punto decimal tipo 2.6 / 5.05 ; si no, miles tipo 28.000 / 1.380
        if len(partes) == 2 and len(partes[1]) <= 2 and len(partes[0]) <= 3:
            pass  # se deja el punto decimal
        else:
            s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def _nums(texto):
    vals = []
    t = texto.lower()
    for m in _NUM_RE.findall(t):
        v = _norm_num(m)
        if v is not None:
            vals.append(v)
    for w, v in _PALABRA_NUM.items():
        if re.search(r"\b" + re.escape(w) + r"\b", t):
            vals.append(float(v))
    return vals


def _tiene(lista, texto):
    t = texto.lower()
    return any(p in t for p in lista)


class Guardian:
    def _riesgo_oracion(self, oracion, frags, fuentes_txt, fuente_citada_txt):
        max_resp = max(oracion.get("respaldo_por_fuente", []) or [0.0])
        max_contra = max(oracion.get("contradiccion_por_fuente", []) or [0.0])
        sim_max = max(oracion.get("similitud_por_fuente", []) or [0.0])
        llm = oracion.get("llm") or {}
        veredicto = llm.get("veredicto")
        conf = llm.get("confianza", 0.0) or 0.0
        texto = oracion.get("texto", "")

        # Rescate multifuente: si cada fragmento esta respaldado por
        # alguna fuente, la oracion que combina fuentes es fiel.
        if frags:
            frag_min = min(
                (max(f.get("respaldo_por_fuente", []) or [0.0]) for f in frags),
                default=max_resp,
            )
            efectivo = max(max_resp, frag_min)
        else:
            efectivo = max_resp
        riesgo = 1.0 - efectivo
        tipo = None

        # LLM: bandera explicita.
        if veredicto in ("no_respaldada", "contradicha"):
            riesgo = max(riesgo, 0.55 + 0.4 * conf)
            tipo = "contradiccion" if veredicto == "contradicha" else "no_respaldado"
        elif veredicto is None:
            riesgo = max(riesgo, 0.70)
        else:  # respaldada -> solo creer si es firme
            if conf < 0.85:
                riesgo = max(riesgo, 0.80)
                tipo = tipo or "no_respaldado"
            elif conf < 0.95:
                riesgo = max(riesgo, 0.55)
            if max_contra > 0.7:
                riesgo = max(riesgo, 0.75)
            # cita una fuente que no se parece: sospechoso
            idx = llm.get("fuente")
            sims = oracion.get("similitud_por_fuente", []) or []
            if isinstance(idx, int) and 0 <= idx < len(sims) and sims[idx] < 0.45:
                riesgo = max(riesgo, 0.70)

        # Chequeo de numeros: numero en respuesta ausente en fuentes.
        nums_resp = _nums(texto)
        if nums_resp:
            nums_fuente = []
            for f in fuentes_txt:
                nums_fuente.extend(_nums(f))
            for v in nums_resp:
                if not nums_fuente:
                    riesgo = max(riesgo, 0.88)
                    tipo = "distorsion"
                    break
                if not any(abs(v - n) < 1e-9 for n in nums_fuente):
                    # Aritmetica permitida: suma/resta/doble con cifras de fuentes
                    ok = False
                    for i, a in enumerate(nums_fuente):
                        if abs(v - 2 * a) < 1e-9:
                            ok = True
                            break
                        for b in nums_fuente[i + 1:]:
                            if abs(v - (a + b)) < 1e-9 or abs(v - abs(a - b)) < 1e-9:
                                ok = True
                                break
                    if not ok:
                        riesgo = max(riesgo, 0.88)
                        tipo = "distorsion"
                        break

        # Alcance: el mismo universal de la respuesta ausente en fuentes.
        # (el NLI es ciego aqui: aunque de respaldo alto, subir)
        univ_resp = [p for p in _UNIVERSALES if p in texto.lower()]
        if univ_resp and not any(
            p in f.lower() for f in fuentes_txt for p in univ_resp
        ):
            riesgo = max(riesgo, 0.78 if max_resp > 0.5 else 0.85)
            tipo = tipo or "sobregeneralizacion"

        # Causa/consecuencia inventada: el conector no esta en ninguna fuente.
        caus_resp = [p for p in _CAUSALES if p in texto.lower()]
        if caus_resp and not any(
            p in f.lower() for f in fuentes_txt for p in caus_resp
        ):
            if efectivo < 0.7:
                riesgo = max(riesgo, 0.85)
                tipo = tipo or "no_respaldado"

        # Rescate de parafrasis fiel: LLM seguro + parecido alto
        # perdona un NLI medio, salvo bandera de texto o NLI hundido.
        if (
            veredicto == "respaldada"
            and conf >= 0.95
            and sim_max >= 0.75
            and tipo is None
            and max_contra < 0.6
            and efectivo >= 0.30
        ):
            riesgo = min(riesgo, 0.38)

        return min(1.0, max(0.0, riesgo)), tipo

    def score(self, caso) -> float:
        oraciones = (caso.get("senales") or {}).get("oraciones", []) or []
        fragmentos = (caso.get("senales") or {}).get("fragmentos", []) or []
        fuentes = caso.get("fuentes", []) or []
        if not oraciones:
            return 0.0
        por_ora = {}
        for f in fragmentos:
            por_ora.setdefault(f.get("oracion", 0), []).append(f)
        peor = 0.0
        for i, o in enumerate(oraciones):
            llm = o.get("llm") or {}
            idx = llm.get("fuente")
            citada = ""
            if isinstance(idx, int) and 0 <= idx < len(fuentes):
                citada = fuentes[idx]
            r, _ = self._riesgo_oracion(o, por_ora.get(i, []), fuentes, citada)
            peor = max(peor, r)
        # Calibracion agresiva: un escape pesa 4x, el corte efectivo
        # baja de 0.5 a ~0.44 en riesgo bruto.
        final = 0.06 + 0.94 * peor
        return min(1.0, max(0.0, final))
