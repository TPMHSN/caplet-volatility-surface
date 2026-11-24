# Du classeur au prix d’un caplet

Un caplet est une option sur le taux d’un coupon futur. Un cap regroupe plusieurs de ces options. Le moteur effectue le trajet inverse à partir des cotations de caps pour obtenir une grille par date de fixing et par strike.

## 1. Calculer le taux à terme du coupon

La courbe €STR fournit le facteur d’actualisation à la date de paiement. La courbe de projection de l’index fournit deux facteurs, au début et à la fin de la période sous-jacente.

$$
F = \frac{P_{\mathrm{proj}}(T_0)/P_{\mathrm{proj}}(T_1)-1}{\tau_{\mathrm{index}}}
$$

Les dates $T_0$ et $T_1$ suivent la convention de l’index. La fraction d’année utilisée pour projeter le taux peut différer de celle utilisée pour pondérer le paiement du contrat. Voir [le calcul du forward](../moteur/services/latest_normal_coupon.py) et [la construction des courbes](../moteur/curves/yield_curve_builder.py).

## 2. Retrouver les vols des coupons

À strike fixé, une cotation de cap se convertit en prime en appliquant la vol cotée à ses coupons. Pour deux caps de même strike dont les échéanciers sont emboîtés, leur écart de prime isole le bloc de coupons supplémentaire. On estime la vol de ce bloc en cherchant celle qui reproduit cette prime. Ce principe donne un point de départ au stripping par maturité.

Le papier de Hagan et Konikov décrit plusieurs familles de stripping, dont les blocs constants et une construction quadratique lissée. Le moteur travaille ici en volatilité **normale**. [Hagan et Konikov, *Interest Rate Volatility Cube: Construction and Use*, section 3 (2004)](https://citeseerx.ist.psu.edu/document?doi=16e6054b918a1facbb1fed303d889865e8b0cfba&repid=rep1&type=pdf)

- **3M.** Le candidat `N_variance_wing` utilise une courbe de variance commune pour les premiers caplets du cap 1 an. Les caplets suivants répartissent les primes résiduelles des piliers 18 mois et 2 ans. Le smile est évalué selon le temps et la moneyness $K-F$. [Code de la calibration 3M](../moteur/services/latest_normal_candidate.py)
- **6M.** Les cibles sont lissées en temps de paiement par une construction quadratique. Le raccord court s’appuie sur le candidat 3M à même moneyness absolue. Le smile 6M combine l’aile D.4 et une cubique naturelle sur les strikes natifs. [Préparation du modèle et cibles de pricing](../moteur/services/latest_normal_model.py)

## 3. Prolonger le 3M à partir du 6M

Au-delà du dernier fixing couvert nativement en 3M, on utilise le fait qu’une période de six mois peut se décomposer en deux sous-périodes de trois mois. La linéarisation du taux composé donne deux poids $w_1$ et $w_2$. Avec une corrélation $\rho$, le facteur de transfert est

$$
A=\sqrt{w_1^2+w_2^2+2\rho w_1 w_2}.
$$

Le modèle lit le smile 6M au strike $K_6=F_6+A(K_3-F_3)$ et déduit $\sigma_3=\sigma_6/A$. Un raccord progressif sur un an relie ce transfert à la dernière coupe native. L’hypothèse pose des amplitudes de vol égales sur les deux sous-périodes et une base déterministe. [Code du prolongement](../moteur/services/latest_normal_extension.py)

## 4. Valoriser chaque coupon

Pour un caplet futur de notionnel $N$, l’annuité vaut $N\tau_{\mathrm{contrat}}P_{\mathrm{disc}}(T_{\mathrm{paiement}})$. En notant $s=\sigma_N\sqrt{T_{\mathrm{fixing}}}$ et $d=(F-K)/s$, sa valeur est

$$
N\tau_{\mathrm{contrat}}P_{\mathrm{disc}}(T_{\mathrm{paiement}})
\left[(F-K)\Phi(d)+s\phi(d)\right].
$$

Le floorlet utilise le put correspondant. Si le fixing est déjà observé, le paiement devient déterministe et le moteur actualise le payoff intrinsèque. Enfin, le prix du cap ou du floor est la somme des valeurs de ses coupons. Le pricer appelle la surface aux dates et au strike du contrat. Il utilise les forwards et strikes d’origine dans Bachelier.

Pour suivre ce trajet dans le code, démarrer dans [replay](../reproduire.py). Les trois appels centraux sont `prepare_candidate_workbook`, `extended_surface` et `price_extended_model`. Le premier produit la calibration et les deux suivants l’utilisent.
