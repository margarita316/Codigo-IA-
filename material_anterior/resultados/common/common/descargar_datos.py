from pathlib import Path

from tdc.multi_pred import DTI

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

for name in ("DAVIS", "KIBA"):
    print("Descargando " + name + " real via PyTDC...")
    dataset = DTI(name=name, path=str(DATA_DIR))
    df = dataset.get_data()
    print("  " + name + ": " + str(len(df)) + " pares, "
          + str(df["Drug_ID"].nunique()) + " farmacos, "
          + str(df["Target_ID"].nunique()) + " proteinas")

print("Descarga completa.")
