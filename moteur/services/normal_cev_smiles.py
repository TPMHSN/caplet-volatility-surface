"""Positive normal strike smiles: natural cubic reference and CEV nu=0 seed.

CEV uses the Hagan B.69 normal approximation, not exact absorbed-CEV prices.
No Black quotation/conversion, market writes, or full SABR fit occurs here.
"""
from math import isfinite, log, sqrt
from typing import Literal
import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares

class NormalCevRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    normal_curves: dict
    time_method: Literal["constant","linear","quadratic"]="quadratic"
    weighting: Literal["paper","uniform","vega"]="paper"
    atm_anchor: bool=True

def _targets(strikes,vols):
    k=np.asarray(strikes,dtype=float); v=np.asarray(vols,dtype=float)
    if k.ndim!=1 or v.shape!=k.shape or len(k)<2 or len(k)>64:
        raise ValueError("Au moins deux strikes et autant de volatilités normales sont requis.")
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(v)) or np.any(k>.05+1e-12) or np.any(np.diff(k)<=0) or np.any(v<=0):
        raise ValueError("Strikes finis croissants jusqu’à 5 % et vols normales positives finies requis.")
    return k,v

def _positive_targets(strikes,vols):
    k,v=_targets(strikes,vols)
    if np.any(k<=0):
        raise ValueError("CEV/SABR sans décalage exige des strikes strictement positifs.")
    return k,v

def normal_strike_vol(strikes,vols,strike):
    """Natural cubic in normal volatility; full observed signed K interval only."""
    k,v=_targets(strikes,vols); query=np.asarray(strike,dtype=float)
    if not np.all(np.isfinite(query)) or np.any(query<k[0]) or np.any(query>k[-1]):
        raise ValueError("Le strike doit rester entre les strikes importés ; aucune extrapolation.")
    out=CubicSpline(k,v,bc_type="natural",extrapolate=False)(query)
    if not np.all(np.isfinite(out)) or np.any(out<=0):
        raise ValueError("La cubique normale devient non positive ; ne pas utiliser ce smile.")
    return float(out) if out.ndim==0 else out

def cev_normal_vol(forward,strikes,expiry,alpha,beta):
    """Hagan normal B.69 at nu=0, rho immaterial, no model displacement."""
    k=np.asarray(strikes,dtype=float)
    if not all(isfinite(float(x)) for x in (forward,expiry,alpha,beta)) or forward<=0 or expiry<=0 or alpha<=0 or not 0<=beta<=1 or not np.all(np.isfinite(k)) or np.any(k<=0):
        raise ValueError("CEV exige F, K, T, alpha > 0 et 0 <= beta <= 1.")
    ell=np.log(forward/k); g=forward*k; b1=1-beta
    numerator=1+ell**2/24+ell**4/1920
    denominator=1+b1*b1*ell**2/24+b1**4*ell**4/1920
    out=alpha*g**(beta/2)*numerator/denominator*(1-beta*(2-beta)*alpha*alpha*expiry/(24*g**b1))
    if not np.all(np.isfinite(out)) or np.any(out<=0):
        raise ValueError("Approximation CEV normale non positive ou non finie.")
    return float(out) if out.ndim==0 else out

def _alpha_atm(forward,expiry,beta,target):
    if beta<1e-14:return target
    c=beta*(2-beta)*expiry/(24*forward**(2*(1-beta)))
    peak=sqrt(1/(3*c))
    def residual(a):return a*forward**beta*(1-c*a*a)-target
    if residual(peak)<0:raise ValueError("Aucune petite racine alpha compatible avec l’ATM Hagan CEV.")
    return brentq(residual,1e-15,peak,xtol=1e-14)

def fit_cev(forward,expiry,strikes,target,weighting="paper",atm_anchor=False):
    k,v=_positive_targets(strikes,target)
    if not isfinite(forward) or forward<=0 or not isfinite(expiry) or expiry<=0:
        raise ValueError("Forward CEV et temps de fixing strictement positifs requis.")
    if weighting=="paper":weights=1/np.sqrt(1+(k-forward)**2)
    elif weighting=="uniform":weights=np.ones(len(k))
    elif weighting=="vega":
        z=(forward-k)/(v*sqrt(expiry))
        weights=np.exp(-.5*z*z)*sqrt(expiry)
        if np.count_nonzero(weights>1e-14)<2:raise ValueError("Moins de deux strikes ont une vega suffisante pour calibrer CEV.")
    else:raise ValueError("Pondération CEV inconnue.")
    weights=weights/weights.sum()
    atm_target=normal_strike_vol(k,v,forward) if k[0]<=forward<=k[-1] else None
    if atm_anchor and atm_target is None:raise ValueError("L’ATM est hors du domaine de la cubique ; aucune extrapolation ATM.")
    level=float(np.dot(weights,v)); starts=[];candidates=[]
    def unpack(p):
        beta=float(p[0] if atm_anchor else p[1])
        alpha=_alpha_atm(forward,expiry,beta,atm_target) if atm_anchor else float(np.exp(p[0]))
        return alpha,beta
    def residual(p):
        try:
            a,b=unpack(p)
            return np.sqrt(weights)*(cev_normal_vol(forward,k,expiry,a,b)-v)*1e4
        except (ValueError,FloatingPointError,OverflowError):return np.full(len(k),1e6)
    for b in (0.,.5,1.):
        try:
            a=_alpha_atm(forward,expiry,b,atm_target) if atm_anchor else level/forward**b
        except ValueError:
            continue
        starts.append({"alpha":a,"beta":b})
        initial=[b] if atm_anchor else [log(a),b]
        bounds=([0.],[1.]) if atm_anchor else ([log(1e-9),0.],[log(1000.),1.])
        solved=least_squares(residual,initial,bounds=bounds,max_nfev=300,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        if solved.success and np.all(np.isfinite(solved.fun)) and np.max(np.abs(solved.fun))<1e6:candidates.append(solved)
    if not candidates:raise ValueError("Calibration CEV non convergée.")
    best=min(candidates,key=lambda r:float(np.dot(r.fun,r.fun)))
    a,b=unpack(best.x);modeled=cev_normal_vol(forward,k,expiry,a,b);errors=(modeled-v)*1e4
    atm=cev_normal_vol(forward,forward,expiry,a,b)
    return {"alpha":a,"beta":b,"nu":0.,"rho":0.,"weights":weights.tolist(),
        "modeled_normal_vols":modeled.tolist(),"rmse_normal_bp":float(np.sqrt(np.mean(errors**2))),
        "max_error_normal_bp":float(np.max(abs(errors))),"fit_cost":float(np.dot(best.fun,best.fun)),
        "atm_target_normal_vol":atm_target,"atm_cev_normal_vol":atm,
        "atm_error_bp":None if atm_target is None else (atm-atm_target)*1e4,
        "bound_hit":b<1e-5 or b>1-1e-5,"initial_parameters":starts}

def build_normal_cev_study(normal_curves,time_method="quadratic",weighting="paper",atm_anchor=True):
    if time_method not in ("constant","linear","quadratic"):raise ValueError("Interpolation temporelle inconnue.")
    if not isinstance(normal_curves,dict) or normal_curves.get("volatility_type")!="normal":
        raise ValueError("Calculer les courbes normales avant CEV.")
    families=normal_curves.get("families",[])
    if not isinstance(families,list) or not 1<=len(families)<=2:raise ValueError("Familles normales manquantes.")
    results=[]
    for family in families:
        tenor=family.get("index_tenor")
        if tenor not in ("3M","6M"):raise ValueError("Indice CEV inconnu.")
        all_curves=sorted(family.get("curves",[]),key=lambda c:c["strike"])
        curves=[c for c in all_curves if c["strike"]>0]
        excluded=[{"strike":c["strike"],"reason":"CEV sans décalage : strike nul ou négatif hors du domaine du modèle."} for c in all_curves if c["strike"]<=0]
        if not 2<=len(curves)<=64:raise ValueError("Au moins deux courbes de strikes sont requises.")
        strikes=[c["strike"] for c in curves]
        _targets(strikes,[.01]*len(strikes))
        flowmaps=[]
        for curve in curves:
            if not curve.get("intervals"):raise ValueError("Courbe normale incomplète : calculer tous les strikes.")
            flows=curve["intervals"][-1]["cashflows"]
            if not 1<=len(flows)<=160:raise ValueError("Nombre de caplets hors limite.")
            flowmaps.append({(f["fixing_date"],f["payment_date"]):f for f in flows})
        if any(set(m)!=set(flowmaps[0]) for m in flowmaps):raise ValueError("Les calendriers des strikes ne correspondent pas.")
        slices=[]
        for key,flow in flowmaps[0].items():
            targets=[m[key]["normal_vols"][time_method] for m in flowmaps]
            row={name:flow[name] for name in ("fixing_date","start_date","end_date","payment_date","expiry_years","forward_rate","accrual_fraction","discount_factor")}
            from datetime import date
            row["maturity_years"]=(date.fromisoformat(row["payment_date"])-date.fromisoformat(normal_curves["valuation_date"])).days/365
            row.update({"strikes":strikes,"target_normal_vols":targets,"status":"unavailable","issues":[],"alpha":None,"beta":None,"nu":0.,"rho":0.,"weights":[],"rmse_normal_bp":None,"max_error_normal_bp":None,"atm_target_normal_vol":None,"atm_cev_normal_vol":None,"atm_error_bp":None,"bound_hit":False,"observations":[],"curve":[],"initial_parameters":[],"fit_cost":None})
            try:
                if any(x is None for x in targets):raise ValueError("Une volatilité temporelle cible est indisponible.")
                fit=fit_cev(row["forward_rate"],row["expiry_years"],strikes,targets,weighting,atm_anchor)
                modeled=fit.pop("modeled_normal_vols");row.update(fit);row["status"]="ready"
                row["observations"]=[{"strike":k,"target_normal_vol":v,"cev_normal_vol":c,"error_normal_bp":(c-v)*1e4,"cubic_normal_vol":v} for k,v,c in zip(strikes,targets,modeled)]
                grid=np.unique(np.r_[np.linspace(strikes[0],strikes[-1],101),strikes,row["forward_rate"] if strikes[0]<=row["forward_rate"]<=strikes[-1] else []])
                cub=normal_strike_vol(strikes,targets,grid)
                cev=cev_normal_vol(row["forward_rate"],grid,row["expiry_years"],row["alpha"],row["beta"])
                row["curve"]=[{"strike":float(k),"cubic_normal_vol":float(cv),"cev_normal_vol":float(v)} for k,cv,v in zip(grid,cub,cev)]
            except (ValueError,TypeError,OverflowError,FloatingPointError) as exc:
                row["status"]="unavailable";row["issues"].append(str(exc))
            slices.append(row)
        results.append({"index_tenor":tenor,"slices":slices,"excluded_strikes":excluded})
    return {"model":"cev_hagan_normal","model_shift":0.,"volatility_type":"normal",
        "valuation_date":normal_curves.get("valuation_date"),"source_token":normal_curves.get("source_token"),
        "time_method":time_method,"weighting":weighting,"atm_anchor":atm_anchor,"families":results,
        "assumptions":[
            "Cibles : volatilités normales des caplets issues des courbes en temps, aux strikes positifs importés. Aucun ajustement aux références ICE.",
            "Cubique naturelle en strike sur la volatilité normale ; domaine des strikes observés seulement, sans extrapolation.",
            "CEV : formule normale asymptotique Hagan B.69 avec nu=0 ; alpha et beta ajustés. Rho est sans effet et sérialisé à zéro.",
            "0 <= beta <= 1 est notre domaine de modèle ; le paragraphe CEV du document ne fixe pas cette borne.",
            "Poids du document : 1/sqrt(1+(K-F)^2), taux en décimal. Poids uniformes et vega normale sont des variantes explicites.",
            "Ancrage ATM optionnel : petite racine positive de l’équation ATM Hagan ; cible cubique du caplet, pas vol flat ATM du cap.",
            "La convergence CEV ne prouve pas la qualité du smile ; contrôler RMSE, erreur ATM et paramètres aux bornes.",
            "Aucune conversion Black, aucun shift, aucune calibration SABR complète."]}
