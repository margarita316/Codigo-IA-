"""Descriptores fisicoquimicos compartidos para farmaco (SMILES) y proteina
(secuencia). El MISMO codigo se usa desde los 3 metodos: lo unico que cambia
entre metodos es la arquitectura del modelo que consume estos vectores.

Farmaco: todos los descriptores 2D de RDKit (rdkit.Chem.Descriptors),
~210 propiedades fisicoquimicas/topologicas/de fragmentos.

Proteina: composicion de aminoacidos (AAC, 20), composicion de dipeptidos
(DPC, 400), descriptores CTD (Composition-Transition-Distribution sobre 7
propiedades fisicoquimicas clasicas de Dubchak, 21 x 7 = 147) y propiedades
globales via Bio.SeqUtils.ProtParam (~10). Total proteina ~= 577 columnas.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

_DESC_FUNCS = Descriptors._descList  # lista de (nombre, funcion)
DRUG_DESCRIPTOR_NAMES = [name for name, _ in _DESC_FUNCS]

MORGAN_BITS = 512
MORGAN_RADIUS = 2

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Clasificacion de aminoacidos en 3 grupos (polar/neutro/hidrofobico, etc.)
# para cada una de las 7 propiedades fisicoquimicas de Dubchak et al. 1995,
# usada tambien en herramientas como iFeature / PROFEAT / PseAAC-Builder.
CTD_GROUPS = {
    "hydrophobicity": (
        "RKEDQN", "GASTPHY", "CLVIMFW",
    ),
    "normalized_vdw_volume": (
        "GASTPDC", "NVEQIL", "MHKFRYW",
    ),
    "polarity": (
        "LIFWCMVY", "PATGS", "HQRKNED",
    ),
    "polarizability": (
        "GASDT", "CPNVEQIL", "KMHFRYW",
    ),
    "charge": (
        "KR", "ANCQGHILMFPSTWYV", "DE",
    ),
    "secondary_structure": (
        "EALMQKRH", "VIYCWFT", "GNPSD",
    ),
    "solvent_accessibility": (
        "ALFCGIVW", "RKQEND", "MPSTHY",
    ),
}


def drug_descriptors(smiles: str) -> np.ndarray:
    """Vector de ~210 descriptores fisicoquimicos 2D de RDKit. NaN si el
    SMILES no es valido o algun descriptor falla numericamente (se limpia
    despues, solo con estadisticas del fold de entrenamiento)."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.full(len(_DESC_FUNCS), np.nan, dtype=np.float64)
    values = []
    for _, func in _DESC_FUNCS:
        try:
            v = func(mol)
            v = float(v)
            if not np.isfinite(v):
                v = np.nan
        except Exception:
            v = np.nan
        values.append(v)
    return np.asarray(values, dtype=np.float64)


def morgan_fingerprint(smiles: str, n_bits: int = MORGAN_BITS) -> np.ndarray:
    """Fingerprint circular ECFP4 (Morgan, radio=2): captura subestructuras
    quimicas que los descriptores escalares de RDKit no ven directamente.
    Vector de NaN si el SMILES no es valido."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.full(n_bits, np.nan, dtype=np.float64)
    bitvec = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float64)
    for i in bitvec.GetOnBits():
        arr[i] = 1.0
    return arr


def drug_descriptors_v2(smiles: str) -> np.ndarray:
    """v1 (RDKit ~217) + Morgan fingerprint (512) = ~729 columnas."""
    return np.concatenate([drug_descriptors(smiles), morgan_fingerprint(smiles)])


def _aac(sequence: str) -> np.ndarray:
    n = len(sequence)
    if n == 0:
        return np.zeros(len(AMINO_ACIDS), dtype=np.float64)
    counts = np.array([sequence.count(aa) for aa in AMINO_ACIDS], dtype=np.float64)
    return counts / n


def _dpc(sequence: str) -> np.ndarray:
    dipeptides = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]
    n = len(sequence) - 1
    vec = np.zeros(len(dipeptides), dtype=np.float64)
    if n <= 0:
        return vec
    idx = {dp: i for i, dp in enumerate(dipeptides)}
    for i in range(len(sequence) - 1):
        dp = sequence[i:i + 2]
        j = idx.get(dp)
        if j is not None:
            vec[j] += 1.0
    return vec / n


def _group_of(residue: str, groups: tuple[str, str, str]) -> int:
    for gi, members in enumerate(groups):
        if residue in members:
            return gi
    return -1


def _ctd_one_property(sequence: str, groups: tuple[str, str, str]) -> np.ndarray:
    n = len(sequence)
    if n == 0:
        return np.zeros(3 + 3 + 15, dtype=np.float64)
    group_ids = np.array([_group_of(r, groups) for r in sequence])
    valid = group_ids >= 0
    group_ids = group_ids[valid]
    n_valid = max(len(group_ids), 1)

    # Composition: fraccion de residuos en cada uno de los 3 grupos.
    composition = np.array([(group_ids == g).sum() for g in range(3)], dtype=np.float64) / n_valid

    # Transition: fraccion de posiciones consecutivas que cambian de grupo (para cada uno de los 3 pares).
    transitions = np.zeros(3, dtype=np.float64)
    if len(group_ids) > 1:
        pairs = list(zip(group_ids[:-1], group_ids[1:]))
        denom = max(len(pairs), 1)
        transitions[0] = sum(1 for a, b in pairs if {a, b} == {0, 1}) / denom
        transitions[1] = sum(1 for a, b in pairs if {a, b} == {0, 2}) / denom
        transitions[2] = sum(1 for a, b in pairs if {a, b} == {1, 2}) / denom

    # Distribution: posicion relativa (0,25,50,75,100 percentil) de cada grupo, para cada uno de los 3 grupos = 15 valores.
    distribution = np.zeros(15, dtype=np.float64)
    for g in range(3):
        positions = np.where(group_ids == g)[0]
        if len(positions) == 0:
            continue
        percentiles = [0, 25, 50, 75, 100]
        for pi, p in enumerate(percentiles):
            rank = int(np.ceil(p / 100.0 * (len(positions) - 1)))
            rank = min(max(rank, 0), len(positions) - 1)
            distribution[g * 5 + pi] = (positions[rank] + 1) / n_valid

    return np.concatenate([composition, transitions, distribution])


def _ctd(sequence: str) -> np.ndarray:
    parts = [_ctd_one_property(sequence, groups) for groups in CTD_GROUPS.values()]
    return np.concatenate(parts)


def _global_properties(sequence: str) -> np.ndarray:
    """Propiedades globales via Bio.SeqUtils.ProtParam. Se limpia la
    secuencia a los 20 aminoacidos estandar antes de calcular (ProtParam
    no acepta caracteres ambiguos como X, B, Z, U, O)."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    clean = "".join(ch for ch in sequence if ch in AMINO_ACIDS)
    n = len(sequence)
    if len(clean) < 2:
        return np.full(11, np.nan, dtype=np.float64)
    try:
        analysis = ProteinAnalysis(clean)
        helix, turn, sheet = analysis.secondary_structure_fraction()
        gravy = analysis.gravy()
        return np.array([
            float(n),
            analysis.molecular_weight(),
            analysis.aromaticity(),
            analysis.instability_index(),
            analysis.isoelectric_point(),
            gravy,
            helix, turn, sheet,
            analysis.charge_at_pH(7.0),
            len(clean) / max(n, 1),
        ], dtype=np.float64)
    except Exception:
        return np.full(11, np.nan, dtype=np.float64)


def protein_descriptors(sequence: str) -> np.ndarray:
    """Vector concatenado AAC(20) + DPC(400) + CTD(147) + globales(11) = 578 dims."""
    sequence = str(sequence).strip().upper()
    return np.concatenate([
        _aac(sequence),
        _dpc(sequence),
        _ctd(sequence),
        _global_properties(sequence),
    ])


def _positional_thirds(sequence: str) -> np.ndarray:
    """AAC calculada por separado en el tercio N-terminal, medio y
    C-terminal de la secuencia (3 x 20 = 60 dims). AAC/DPC/CTD son todas
    'bolsa de residuos' -- pierden en que parte de la proteina esta cada
    aminoacido; esto recupera algo de esa informacion posicional sin
    necesitar la secuencia completa token a token."""
    n = len(sequence)
    if n < 3:
        return np.zeros(3 * len(AMINO_ACIDS), dtype=np.float64)
    third = n // 3
    partes = [sequence[:third], sequence[third:2 * third], sequence[2 * third:]]
    return np.concatenate([_aac(p) for p in partes])


PROTEIN_DESCRIPTOR_DIM = len(AMINO_ACIDS) + len(AMINO_ACIDS) ** 2 + 7 * (3 + 3 + 15) + 11
DRUG_DESCRIPTOR_DIM = len(DRUG_DESCRIPTOR_NAMES)


def protein_descriptors_v2(sequence: str) -> np.ndarray:
    """v1 (578) + composicion por tercios (60) = 638 columnas."""
    sequence = str(sequence).strip().upper()
    return np.concatenate([protein_descriptors(sequence), _positional_thirds(sequence)])


# Metadatos de "bloques con significado real" (regla del usuario: en vez de
# cortar el vector en pedazos arbitrarios, cada metodo que necesite tratar
# la entrada como una secuencia de "tokens" usa estos bloques, que SI tienen
# un significado coherente (cada uno es un tipo de descriptor distinto).
# Formato: lista de (nombre, tamano). El orden coincide exactamente con el
# orden de concatenacion en drug_descriptors_v2 / protein_descriptors_v2.
BLOQUES_FARMACO_V2 = [
    ("rdkit_descriptores", DRUG_DESCRIPTOR_DIM),
    ("morgan_fingerprint", MORGAN_BITS),
]

BLOQUES_PROTEINA_V2 = [
    ("aac", len(AMINO_ACIDS)),
    ("dpc", len(AMINO_ACIDS) ** 2),
    ("ctd", 7 * (3 + 3 + 15)),
    ("globales", 11),
    ("posicional_tercios", 3 * len(AMINO_ACIDS)),
]

DRUG_DESCRIPTOR_V2_DIM = sum(size for _, size in BLOQUES_FARMACO_V2)
PROTEIN_DESCRIPTOR_V2_DIM = sum(size for _, size in BLOQUES_PROTEINA_V2)


def rangos_de_bloques(bloques: list[tuple[str, int]]) -> dict[str, tuple[int, int]]:
    """Convierte [(nombre, tamano), ...] en {nombre: (inicio, fin)} sobre
    el vector concatenado, para poder recortar cada bloque despues."""
    rangos = {}
    inicio = 0
    for nombre, tamano in bloques:
        rangos[nombre] = (inicio, inicio + tamano)
        inicio += tamano
    return rangos
