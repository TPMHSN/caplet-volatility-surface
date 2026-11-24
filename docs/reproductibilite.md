# Reproductibilité

Les figures du README proviennent du rejeu du 20 novembre 2026. Pour les recréer, lancer le calcul avec le classeur et les historiques de cette date, en ajoutant `--pricing examples/pricing.json`.

```sh
python -m pip install -r requirements-dev.txt
python tools/generer_figures.py CHEMIN_DU_RAPPORT visuels
python -m pytest -q
```

`CHEMIN_DU_RAPPORT` est le dossier contenant `surfaces.json`, `calibration.json` et `pricing.json`. Le script de figures relit ces sorties. Le calcul des forwards et des vols se fait en amont dans `reproduire.py`.

Le rejeu publié couvre 119 dates 3M et 59 dates 6M avec 14 strikes par date. Les contrôles sur le classeur de référence ont vérifié le calcul des forwards par ratio de facteurs de projection, les premiers coupons déjà fixés et la parité cap–floor 3M. Les tests rapides du dépôt vérifient le chargement des fixings, la convention de date et la cohérence d’un prix normal.

## Régénérer la démo 3D

La [démo dans le navigateur](https://tpmhsn.github.io/caplet-volatility-surface/) est publiée avec GitHub Pages. Pour générer le même aperçu à partir d’un rejeu local, exécuter cette commande depuis la racine du dépôt.

```sh
python tools/publier_apercu.py CHEMIN_DU_RAPPORT
```

La commande crée `docs/index.html`. La page affiche les coordonnées du graphique et les volatilités arrondies à 0,1 bp. Le rapport complet s’ouvre depuis `resultats/` et contient les détails du calcul.
