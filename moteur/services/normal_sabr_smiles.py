"""Direct normal Hagan SABR fit to a natural cubic, without CEV/Black conversion."""
from math import sqrt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from services.normal_cev_smiles import _targets, _positive_targets, normal_strike_vol
from services.caplet_black_sabr import hagan_normal_vol

class NormalSabrRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    normal_curves: dict
    time_method: Literal["constant","linear","quadratic"]="quadratic"
    weighting: Literal["paper","uniform","vega"]="paper"
    atm_anchor: bool=True
    beta: float | None=Field(default=None,ge=0,le=1)

def sabr_normal_vol(F,K,T,alpha,beta,rho,nu):
    k=np.asarray(K,dtype=float)
    out=np.array([hagan_normal_vol(F,float(x),T,alpha,beta,rho,nu) for x in k.flat]).reshape(k.shape)
    return float(out) if out.ndim==0 else out

def alpha_atm(F,T,b,rho,nu,target):
    fb=F**b;scale=F**(1-b)
    coeff=[-fb*T*b*(2-b)/(24*scale**2),fb*T*rho*b*nu/(4*scale),fb*(1+T*(2-3*rho*rho)*nu*nu/24),-target]
    roots=np.roots(np.trim_zeros(coeff,'f'))
    valid=sorted(float(z.real) for z in roots if abs(z.imag)<1e-9 and z.real>0)
    if not valid:raise ValueError("Aucune petite racine positive alpha pour l’ATM SABR.")
    return valid[0]

def policy_weights(k,v,F,T,policy):
    if policy=='paper':w=1/np.sqrt(1+(k-F)**2)
    elif policy=='uniform':w=np.ones(len(k))
    elif policy=='vega':w=np.exp(-.5*((F-k)/(v*sqrt(T)))**2)
    else:raise ValueError("Pondération inconnue.")
    if w.sum()<=1e-14:raise ValueError("Vega insuffisante.")
    return w/w.sum()

def fit_sabr(forward,expiry,strikes,target,weighting="paper",atm_anchor=True,beta=None,grid_size=81):
    k,v=_positive_targets(strikes,target);F=float(forward);T=float(expiry)
    if not np.isfinite(F) or not np.isfinite(T) or F<=0 or T<=0:raise ValueError("F et temps de fixing doivent être positifs.")
    if beta is not None and not 0<=beta<=1:raise ValueError("Beta doit être dans [0,1].")
    atm_target=normal_strike_vol(k,v,F) if k[0]<=F<=k[-1] else None
    if atm_anchor and atm_target is None:raise ValueError("ATM hors des strikes observés ; aucune extrapolation.")
    grid=np.unique(np.r_[np.linspace(k[0],k[-1],grid_size),k,F if atm_target is not None else []])
    cub=normal_strike_vol(k,v,grid)
    widths=np.diff(grid);quad=np.r_[widths[0]/2,(widths[:-1]+widths[1:])/2,widths[-1]/2]
    w=quad*policy_weights(grid,cub,F,T,weighting);w/=w.sum()
    nativew=policy_weights(k,v,F,T,weighting)
    def unpack(p):
        offset=0
        if not atm_anchor:alpha=float(np.exp(p[0]));offset=1
        b=float(p[offset]) if beta is None else float(beta)
        if beta is None:offset+=1
        rho,nu=map(float,p[offset:offset+2])
        if atm_anchor:alpha=alpha_atm(F,T,b,rho,nu,atm_target)
        return alpha,b,rho,nu
    def residual(p):
        try:return np.sqrt(w)*(sabr_normal_vol(F,grid,T,*unpack(p))-cub)*1e4
        except (ValueError,OverflowError,FloatingPointError):return np.full(len(grid),1e6)
    candidates=[];starts=[]
    for b,rho,nu in [(0.,0.,.5),(.5,-.3,.5),(1.,.3,.5),(.5,0.,1.5)]:
        if beta is not None:b=beta
        try:a=alpha_atm(F,T,b,rho,nu,atm_target) if atm_anchor else float(np.mean(v))/F**b
        except ValueError:continue
        starts.append(dict(alpha=a,beta=b,rho=rho,nu=nu))
        p=[];lo=[];hi=[]
        if not atm_anchor:p+=[np.log(a)];lo+=[np.log(1e-9)];hi+=[np.log(100)]
        if beta is None:p+=[b];lo+=[0.];hi+=[1.]
        p+=[rho,nu];lo+=[-.995,0.];hi+=[.995,3.]
        solved=least_squares(residual,p,bounds=(lo,hi),max_nfev=250,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        if solved.success and np.max(abs(solved.fun))<1e5:candidates.append(solved)
    if not candidates:raise ValueError("SABR : aucune initialisation n’a convergé.")
    best=min(candidates,key=lambda s:np.dot(s.fun,s.fun));a,b,rho,nu=unpack(best.x)
    modeled=sabr_normal_vol(F,k,T,a,b,rho,nu);dense=sabr_normal_vol(F,grid,T,a,b,rho,nu)
    e=(modeled-v)*1e4;eg=(dense-cub)*1e4;atm=sabr_normal_vol(F,F,T,a,b,rho,nu)
    return dict(alpha=a,beta=b,rho=rho,nu=nu,weights=nativew.tolist(),modeled_normal_vols=modeled.tolist(),
      rmse_normal_bp=float(np.sqrt(np.mean(e*e))),max_error_normal_bp=float(max(abs(e))),
      cubic_rmse_bp=float(np.sqrt(np.dot(quad,eg*eg)/quad.sum())),cubic_max_error_bp=float(max(abs(eg))),
      atm_target_normal_vol=atm_target,atm_sabr_normal_vol=atm,atm_error_bp=None if atm_target is None else (atm-atm_target)*1e4,
      bound_hit=bool((beta is None and (b<1e-5 or b>1-1e-5)) or abs(rho)>.9949 or nu>2.9999 or nu<1e-5),
      initial_parameters=starts,fit_cost=float(np.dot(best.fun,best.fun)))

def _smile_flowmaps(curves):
    maps=[]
    for curve in curves:
        if not curve.get("intervals"):raise ValueError("Courbe normale incomplète : calculer tous les strikes.")
        flows=curve["intervals"][-1]["cashflows"]
        if not 1<=len(flows)<=160:raise ValueError("Nombre de caplets hors limite.")
        maps.append({(f["fixing_date"],f["payment_date"]):f for f in flows})
    if maps and any(set(m)!=set(maps[0]) for m in maps):
        raise ValueError("Les calendriers des strikes ne correspondent pas.")
    return maps


def _smile_grid(strikes, forward):
    return np.unique(np.r_[np.linspace(strikes[0],strikes[-1],101),strikes,
        forward if strikes[0]<=forward<=strikes[-1] else []])


def build_normal_sabr_study(normal_curves,time_method="quadratic",weighting="paper",atm_anchor=True,beta=None):
    if time_method not in ("constant","linear","quadratic"):raise ValueError("Interpolation temporelle inconnue.")
    if not isinstance(normal_curves,dict) or normal_curves.get("volatility_type")!="normal":
        raise ValueError("Calculer les courbes normales avant SABR.")
    families=normal_curves.get("families",[])
    if not isinstance(families,list) or not 1<=len(families)<=2:raise ValueError("Familles normales manquantes.")
    results=[]
    for family in families:
        tenor=family.get("index_tenor")
        if tenor not in ("3M","6M"):raise ValueError("Indice SABR inconnu.")
        all_curves=sorted(family.get("curves",[]),key=lambda c:c["strike"])
        all_strikes=[c["strike"] for c in all_curves]
        _targets(all_strikes,[.01]*len(all_strikes))
        curves=[c for c in all_curves if c["strike"]>0]
        strikes=[c["strike"] for c in curves]
        exclusions=[{"strike":k,"reason":"SABR Hagan B.69 sans déplacement exige K > 0 ; disponible en cubique normale."}
            for k in all_strikes if k<=0]
        # Each model keeps its own domain. An unavailable negative wing must
        # neither remove an imported quote silently nor disable positive SABR.
        full_issue=None;positive_issue=None
        try:full_maps=_smile_flowmaps(all_curves)
        except (ValueError,KeyError,TypeError) as exc:full_maps=[];full_issue=str(exc)
        try:flowmaps=_smile_flowmaps(curves)
        except (ValueError,KeyError,TypeError) as exc:flowmaps=[];positive_issue=str(exc)
        reference_maps=flowmaps or full_maps
        if not reference_maps:
            available=[c for c in all_curves if c.get("intervals")]
            reference_maps=_smile_flowmaps(available)
        if not reference_maps:raise ValueError("Aucun calendrier normal disponible.")
        if full_maps and set(full_maps[0])!=set(reference_maps[0]):
            full_issue="Les calendriers de la cubique complète ne correspondent pas à SABR."
        slices=[]
        for key,flow in reference_maps[0].items():
            targets=[m[key]["normal_vols"][time_method] for m in flowmaps] if flowmaps else [None]*len(strikes)
            row={name:flow[name] for name in ("fixing_date","start_date","end_date","payment_date","expiry_years","forward_rate","accrual_fraction","discount_factor")}
            from datetime import date
            row["maturity_years"]=(date.fromisoformat(row["payment_date"])-date.fromisoformat(normal_curves["valuation_date"])).days/365
            row.update({"strikes":strikes,"target_normal_vols":targets,"status":"unavailable","issues":[],"alpha":None,"beta":None,"nu":0.,"rho":0.,"weights":[],"cubic_rmse_bp":None,"cubic_max_error_bp":None,"rmse_normal_bp":None,"max_error_normal_bp":None,"atm_target_normal_vol":None,"atm_sabr_normal_vol":None,"atm_error_bp":None,"bound_hit":False,"observations":[],"curve":[],"initial_parameters":[],"fit_cost":None,
                "excluded_strikes":exclusions,"cubic_status":"unavailable","cubic_issues":[],"cubic_curve":[]})
            try:
                if full_issue:raise ValueError(full_issue)
                full_targets=[m[key]["normal_vols"][time_method] for m in full_maps]
                if any(x is None for x in full_targets):raise ValueError("Une volatilité temporelle de la cubique complète est indisponible.")
                full_grid=_smile_grid(all_strikes,row["forward_rate"])
                full_cubic=normal_strike_vol(all_strikes,full_targets,full_grid)
                row["cubic_curve"]=[{"strike":float(k),"cubic_normal_vol":float(v),"sabr_normal_vol":None}
                    for k,v in zip(full_grid,full_cubic)]
                row["cubic_status"]="ready"
                # Check the extended wing separately from the unchanged SABR
                # reference; this diagnoses shape without changing the spline.
                check_grid=np.unique(np.r_[np.linspace(all_strikes[0],all_strikes[-1],1001),all_strikes])
                check_vols=normal_strike_vol(all_strikes,full_targets,check_grid)
                std=check_vols*np.sqrt(row["expiry_years"]);z=(row["forward_rate"]-check_grid)/std
                calls=(row["forward_rate"]-check_grid)*ndtr(z)+std*np.exp(-z*z/2)/np.sqrt(2*np.pi)
                slopes=np.diff(calls)/np.diff(check_grid)
                checks={"monotonicity_violations":int(np.sum(slopes>1e-8)),"convexity_violations":int(np.sum(np.diff(slopes)<-1e-6)),"digital_bound_violations":int(np.sum((slopes>1e-8)|(slopes<-1-1e-8)))}
                row["cubic_price_grid_checks"]=checks
                if any(checks.values()):
                    row["cubic_issues"].append("La cubique complète présente des violations de monotonie, convexité ou bornes digitales sur la grille de contrôle ; examiner ce smile.")
            except (ValueError,KeyError,TypeError,OverflowError,FloatingPointError) as exc:
                row["cubic_issues"].append(str(exc))
            try:
                if positive_issue:raise ValueError(positive_issue)
                if any(x is None for x in targets):raise ValueError("Une volatilité temporelle cible est indisponible.")
                fit=fit_sabr(row["forward_rate"],row["expiry_years"],strikes,targets,weighting,atm_anchor,beta)
                modeled=fit.pop("modeled_normal_vols");row.update(fit);row["status"]="ready"
                row["observations"]=[{"strike":k,"target_normal_vol":v,"sabr_normal_vol":c,"error_normal_bp":(c-v)*1e4,"cubic_normal_vol":v} for k,v,c in zip(strikes,targets,modeled)]
                grid=_smile_grid(strikes,row["forward_rate"])
                cub=normal_strike_vol(strikes,targets,grid)
                modeled_grid=sabr_normal_vol(row["forward_rate"],grid,row["expiry_years"],row["alpha"],row["beta"],row["rho"],row["nu"])
                row["curve"]=[{"strike":float(k),"cubic_normal_vol":float(cv),"sabr_normal_vol":float(v)} for k,cv,v in zip(grid,cub,modeled_grid)]
                std=modeled_grid*np.sqrt(row["expiry_years"]);z=(row["forward_rate"]-grid)/std
                call=(row["forward_rate"]-grid)*ndtr(z)+std*np.exp(-z*z/2)/np.sqrt(2*np.pi)
                slopes=np.diff(call)/np.diff(grid)
                checks={"monotonicity_violations":int(np.sum(slopes>1e-8)),"convexity_violations":int(np.sum(np.diff(slopes)<-1e-6)),"digital_bound_violations":int(np.sum((slopes>1e-8)|(slopes<-1-1e-8)))}
                row["price_grid_checks"]=checks
                if any(checks.values()):row["issues"].append("La grille de prix SABR présente des violations de monotonie, convexité ou bornes digitales : ne pas considérer ce smile comme validé.")
            except (ValueError,TypeError,OverflowError,FloatingPointError) as exc:
                row["status"]="unavailable";row["issues"].append(str(exc))
            slices.append(row)
        results.append({"index_tenor":tenor,"slices":slices})
    return {"model":"sabr_hagan_normal","model_shift":0.,"volatility_type":"normal",
        "valuation_date":normal_curves.get("valuation_date"),"source_token":normal_curves.get("source_token"),
        "time_method":time_method,"weighting":weighting,"atm_anchor":atm_anchor,"beta":beta,"families":results,
        "assumptions":["Hagan normal B.69, sans shift ni conversion Black. Strikes positifs observés uniquement ; K <= 0 disponible en cubique normale, sans prix SABR.",
        "Ajustement SABR direct à la cubique naturelle des seuls strikes positifs par quadrature trapézoïdale ; la cible ATM et la méthode de calibration restent identiques.",
        "La surface cubique complète utilise tous les strikes importés, nuls et négatifs inclus. Elle est distincte de la référence positive de calibration SABR.",
        "Initialisations multiples indépendantes du CEV ; beta libre [0,1], rho [-0.995,0.995], nu [0,3].",
        "ATM cible issue de la cubique du caplet, pas de la cotation flat ATM du cap.",
        "Hagan est asymptotique ; convergence et ancrage ne garantissent ni précision globale ni absence d’arbitrage."]}
