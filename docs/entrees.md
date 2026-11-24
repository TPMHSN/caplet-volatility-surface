# Entrées et conventions

Le programme lit un classeur `.xlsx` et des historiques de fixings locaux. Les cotations doivent être enregistrées sous forme de valeurs lisibles par le parseur du classeur, y compris lorsqu’elles proviennent de formules Excel.

| Feuille | Format | Rôle |
| --- | --- | --- |
| `ESTR` | Colonnes `Term` et `Market Rate` | Courbe d’actualisation en EUR |
| `EUR3` | Colonnes `Term` et `Market Rate` | Courbe de projection Euribor 3M |
| `EUR6` | Colonnes `Term` et `Market Rate` | Courbe de projection Euribor 6M |
| `VOL CAP` | Une ligne par maturité et des strikes absolus en colonnes | Volatilités normales de caps en bp |

Dans les feuilles de courbes, le format de taux du classeur de référence est en pourcentage. Dans `VOL CAP`, un strike de 3 % est affiché en pourcentage et une vol normale de 85 bp est saisie comme `85`. La colonne ATM éventuelle est interprétée selon les conventions du chargeur du moteur. La grille de visualisation par défaut va de −1,5 % à 5 %.

Pour dater le rejeu, employer un fichier nommé `AAAAMMJJ - md.xlsx` ou passer `--date AAAA-MM-JJ`. Le calendrier du moteur exige un jour ouvré TARGET.

## Historiques

Par défaut, le lanceur lit `local/fixings/` à la racine du dépôt. `--fixings-dir` permet de choisir un autre emplacement. Placer dans ce dossier `EUR_3M.json` et `EUR_6M.json`, ainsi que `EUR_OIS.json` si l’historique €STR est disponible.

Exemple de structure avec un taux fictif et une date illustrative.

```json
{
  "currency": "EUR",
  "index_tenor": "3M",
  "fixings": [
    {"date": "2025-01-06", "rate": 0.025}
  ]
}
```

Les taux des historiques sont **décimaux**. Le moteur requiert l’observation Euribor 3M et 6M du jour de marché. Pour le pricing, prévoir aussi les observations des coupons passés qui restent à payer. Avec `--hors-ligne`, les taux proviennent des fichiers fournis. Quand une observation du jour manque, le mode en ligne tente de récupérer la série publique correspondante. Un fixing requis indisponible déclenche un message indiquant l’index et la date. [Séries Euribor de la Bundesbank](https://www.bundesbank.de/en/statistics/money-and-capital-markets/interest-rates-and-yields/money-market-rates-651538) · [Série €STR de la BCE](https://data.ecb.europa.eu/data/datasets/EST/EST.B.EU000A2X2A25.WT)

## Contrats

Dans [l’exemple de pricing](../examples/pricing.json), choisir `cap` ou `floor`, l’index, les dates, le notionnel et le strike. Un strike de 3 % s’écrit `0.03`. Pour un profil amortissable, renseigner `coupon_schedule` avec un objet `{"notional": 10000000, "strike": 0.03}` par coupon, dans l’ordre du calendrier.
