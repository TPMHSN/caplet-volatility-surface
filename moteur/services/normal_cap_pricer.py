"""EUR future caps from normal time curves, cubic strike smile and direct SABR."""
from bisect import bisect_left
from calendar import monthrange
from datetime import date
from math import exp,fsum,pi,sqrt,isfinite
from typing import Literal
import numpy as np
import QuantLib as ql
from scipy.optimize import brentq
from scipy.special import ndtr
from pydantic import Field,FiniteFloat
from curves.helpers import date_to_ql,ql_to_date,ql_evaluation_date
from services.thomas_optionlet import loaded_context
from services.normal_sabr_smiles import NormalSabrRequest,fit_sabr,sabr_normal_vol
from services.normal_cev_smiles import normal_strike_vol,_targets
from services.caplet_normal_curves import _vol_at

class NormalCapRequest(NormalSabrRequest):
    valuation_date: date
    source_token: str=Field(min_length=1)
    notional: FiniteFloat=Field(gt=0,le=1e12)
    effective_date: date
    maturity_date: date
    strike: FiniteFloat
    index_tenor: Literal['3M','6M']
    business_day_convention: Literal['ModifiedFollowing','Following','Preceding','Unadjusted']='ModifiedFollowing'
    day_count: Literal['ACT/360','ACT/365F','30/360']='ACT/360'
    date_generation: Literal['Forward','Backward']='Forward'
    fixing_days: int=Field(default=2,ge=0,le=10)
    payment_lag: int=Field(default=0,ge=0,le=10)
    roll_day: int | None=Field(default=None,ge=1,le=31)
    end_of_month: bool=False

def normal_price_vega(F,K,sigma,T,annuity,notional):
    std=sigma*sqrt(T);z=(F-K)/std;density=exp(-z*z/2)/sqrt(2*pi)
    return notional*annuity*((F-K)*ndtr(z)+std*density),notional*annuity*sqrt(T)*density*1e-4

def target_at(curve,term,method,payment,fixing=None):
    """Payment-time axis retained. Exact fitted polynomial, no time extrapolation."""
    flows=curve.get('intervals',[])[-1]['cashflows'] if curve.get('intervals') else []
    if payment is not None:
        node=next((f for f in flows if f['payment_date']==payment),None)
        if node is None and fixing is not None:node=next((f for f in flows if f['fixing_date']==fixing),None)
        if node is not None:
            sigma=node['normal_vols'][method]
            if sigma is None:raise ValueError('Volatilité temporelle indisponible.')
            return float(sigma),False,False
    fitted=curve.get('time_interpolation')
    if fitted:
        times=fitted['times'];prefix=fitted['prefix']
        lo=prefix[0]['time'] if prefix else times[0]
        if term<lo-1e-10 or term>times[-1]+1e-10:raise ValueError('Paiement hors du domaine de la courbe normale ; aucune extrapolation.')
        if prefix and term<times[0]:
            sigma=float(np.interp(term,[x['time'] for x in prefix],[x['vol'] for x in prefix]))
        elif method=='constant':
            i=min(max(bisect_left(times,term)-1,0),len(times)-2);sigma=float(fitted['constant_vols'][i])
        else:
            parameters=fitted[method+'_parameters']
            if parameters is None:raise ValueError('Interpolation temporelle non calibrée.')
            sigma=float(_vol_at(parameters,times,term,method=='quadratic'))
        if not isfinite(sigma) or sigma<=0:raise ValueError('Volatilité interpolée non positive.')
        return sigma,True,False
    # Old snapshots carry only a plotting mesh; make that approximation explicit.
    samples=curve.get('samples',[])
    if not samples or term<samples[0]['maturity_years']-1e-10 or term>samples[-1]['maturity_years']+1e-10:
        raise ValueError('Paiement hors du domaine de la courbe normale ; aucune extrapolation.')
    values=[x[method] for x in samples]
    if any(v is None for v in values):raise ValueError('Interpolation temporelle indisponible.')
    return float(np.interp(term,[x['maturity_years'] for x in samples],values)),True,True

def cap_schedule(req,calendar,convention):
    if req.effective_date>=req.maturity_date:raise ValueError('La date de fin doit suivre la date de début.')
    if (req.maturity_date-req.effective_date).days>365*40:raise ValueError('Échéancier supérieur à 40 ans.')
    if req.roll_day is None:
        schedule=list(ql.Schedule(date_to_ql(req.effective_date),date_to_ql(req.maturity_date),ql.Period(req.index_tenor),calendar,convention,convention,ql.DateGeneration.Forward if req.date_generation=='Forward' else ql.DateGeneration.Backward,req.end_of_month))
    else:
        raw=[req.effective_date,req.maturity_date];months=3 if req.index_tenor=='3M' else 6
        anchor=req.effective_date if req.date_generation=='Forward' else req.maturity_date;direction=1 if req.date_generation=='Forward' else -1
        for i in range(1,162):
            total=anchor.year*12+anchor.month-1+direction*months*i;y,m=total//12,total%12+1
            d=monthrange(y,m)[1] if req.end_of_month else min(req.roll_day,monthrange(y,m)[1]);point=date(y,m,d)
            if point>=req.maturity_date or point<=req.effective_date:break
            raw.append(point)
        schedule=[calendar.adjust(date_to_ql(d),convention) for d in sorted(raw)]
    if not 2<=len(schedule)<=161 or any(b<=a for a,b in zip(schedule,schedule[1:])):raise ValueError('Échéancier vide, dégénéré ou supérieur à 160 caplets.')
    return schedule

def _unavailable_model(reason, total=False):
    result=dict(status='unavailable',reason=str(reason),price=None,normal_vega_eur_bp=None)
    result.update(dict(unit_price=None,atm_normal_vol=None) if total else dict(normal_vol=None))
    return result


def _caplet_targets(curves,term,payment,fixing,method):
    if len(curves)<2:
        raise ValueError('Au moins deux strikes du domaine du modèle sont requis.')
    if any(not c.get('intervals') or not c['intervals'][-1].get('cashflows') for c in curves):
        raise ValueError('Une courbe de strike requise est indisponible ; recalculer la calibration et examiner ses diagnostics.')
    values=[target_at(c,term,method,payment,fixing=fixing) for c in curves]
    return [v[0] for v in values],values


def price_normal_cap(req,market_store,curve_store):
    context=loaded_context(market_store,curve_store)
    snapshot=req.normal_curves
    if req.source_token!=context.source_token or context.curve_date!=req.valuation_date or snapshot.get('source_token')!=req.source_token or snapshot.get('valuation_date')!=str(req.valuation_date):
        raise ValueError('Sources modifiées : recalculer les courbes normales avec les données chargées à la même date.')
    if snapshot.get('volatility_type')!='normal':raise ValueError('Une surface normale est requise.')
    if not all(c.current for c in context.curves if c.key in ('OIS',req.index_tenor)):raise ValueError('Construire les courbes €STR et Euribor à la même date.')
    family=next((f for f in snapshot.get('families',[]) if f['index_tenor']==req.index_tenor),None)
    if family is None:raise ValueError('Indice absent des courbes normales.')
    if not isinstance(family.get('curves'),list) or not 2<=len(family['curves'])<=64:raise ValueError('Au moins deux courbes normales de strikes sont requises.')
    curves=sorted(family['curves'],key=lambda c:c['strike']);strikes=np.array([c['strike'] for c in curves]);_targets(strikes,np.ones(len(strikes))*.01)
    if not strikes[0]<=req.strike<=strikes[-1]:raise ValueError('Strike hors du domaine observé ; aucune extrapolation.')
    positive_curves=[c for c in curves if c['strike']>0]
    positive_strikes=np.array([c['strike'] for c in positive_curves])
    calendar=ql.TARGET();convention={'ModifiedFollowing':ql.ModifiedFollowing,'Following':ql.Following,'Preceding':ql.Preceding,'Unadjusted':ql.Unadjusted}[req.business_day_convention]
    dc={'ACT/360':ql.Actual360(),'ACT/365F':ql.Actual365Fixed(),'30/360':ql.Thirty360(ql.Thirty360.BondBasis)}[req.day_count]
    curve_set=curve_store.get_curve_set('EUR');projection=curve_set.get_forward_curve(req.index_tenor);discount=curve_set.get_discount_curve('OIS')
    warnings=[];rows=[];cached=[]
    if len(positive_curves)!=len(curves):
        warnings.append('Cubique normale : tous les strikes observés sont utilisés. SABR sans décalage : calibration conservée sur les strikes strictement positifs ; aucun prix SABR aux strikes nuls ou négatifs.')
    with ql_evaluation_date(req.valuation_date):
        valuation=date_to_ql(req.valuation_date);schedule=cap_schedule(req,calendar,convention)
        for start,end in zip(schedule,schedule[1:]):
            fixing=calendar.advance(start,-req.fixing_days,ql.Days);payment=calendar.advance(end,req.payment_lag,ql.Days)
            if fixing<=valuation:raise ValueError(f'Fixing {ql_to_date(fixing)} passé ou du jour : seuls les caplets futurs sont disponibles dans ce pricer.')
            if start<projection.referenceDate() or end>projection.maxDate() or payment>discount.maxDate():raise ValueError('Dates hors du domaine des courbes de taux.')
            index_accrual=ql.Actual360().yearFraction(start,end);F=(projection.discount(start)/projection.discount(end)-1)/index_accrual
            T=ql.Actual365Fixed().yearFraction(valuation,fixing);term=ql.Actual365Fixed().yearFraction(valuation,payment)
            payment_string=str(ql_to_date(payment));fixing_string=str(ql_to_date(fixing))
            available_curve=next((c for c in curves if c.get('intervals') and c['intervals'][-1].get('cashflows')),None)
            if available_curve is not None:
                flows=available_curve['intervals'][-1]['cashflows']
                first_term=(date.fromisoformat(flows[0]['payment_date'])-req.valuation_date).days/365;last_term=(date.fromisoformat(flows[-1]['payment_date'])-req.valuation_date).days/365
                native_fixing=any(f['fixing_date']==fixing_string for f in flows)
                if not native_fixing and (term<first_term-1e-10 or term>last_term+1e-10):raise ValueError('Paiement hors du domaine temporel des caplets ; aucune extrapolation.')
            accrual=dc.yearFraction(start,end);df=discount.discount(payment);A=accrual*df
            models={};issues=[];fit=None;model_cache={};target_values=[]
            for name,model_curves,model_strikes in [('cubic',curves,strikes),('sabr',positive_curves,positive_strikes)]:
                try:
                    if name=='sabr':
                        if req.strike<=0:raise ValueError('SABR Hagan sans décalage exige un strike strictement positif ; la méthode est conservée.')
                        if len(model_strikes)<2:raise ValueError('Au moins deux strikes strictement positifs sont requis pour SABR sans décalage.')
                        if not model_strikes[0]<=req.strike<=model_strikes[-1]:raise ValueError('Strike hors du domaine positif observé SABR ; aucune extrapolation.')
                    targets,values=_caplet_targets(model_curves,term,payment_string,fixing_string,req.time_method)
                    if name=='sabr':
                        fit=fit_sabr(F,T,model_strikes,targets,req.weighting,req.atm_anchor,req.beta)
                        a,b,rho,nu=[fit[key] for key in ('alpha','beta','rho','nu')]
                        vol=sabr_normal_vol(F,req.strike,T,a,b,rho,nu)
                        if fit['bound_hit']:issues.append('Paramètre SABR à une borne de calibration.')
                    else:vol=normal_strike_vol(model_strikes,targets,req.strike)
                    price,vega=normal_price_vega(F,req.strike,vol,T,A,req.notional)
                    if not all(isfinite(x) for x in (vol,price,vega)) or price<0 or vega<0:raise ValueError('Prix ou vega normale non fini ou négatif.')
                    models[name]=dict(status='ready',reason=None,normal_vol=vol,price=price,normal_vega_eur_bp=vega)
                    model_cache[name]=dict(targets=targets,fit=fit if name=='sabr' else None,strikes=model_strikes)
                    target_values.extend(values)
                    grid=np.linspace(model_strikes[0],model_strikes[-1],101)
                    gv=sabr_normal_vol(F,grid,T,a,b,rho,nu) if name=='sabr' else normal_strike_vol(model_strikes,targets,grid)
                    std=gv*sqrt(T);z=(F-grid)/std;calls=(F-grid)*ndtr(z)+std*np.exp(-z*z/2)/sqrt(2*pi);slopes=np.diff(calls)/np.diff(grid)
                    if np.any(np.diff(slopes)<-1e-6) or np.any(slopes>1e-8) or np.any(slopes<-1-1e-8):issues.append(f'{name.upper()} : la grille de prix présente des violations de convexité, monotonie ou bornes digitales.')
                except (ValueError,RuntimeError,OverflowError,FloatingPointError) as exc:
                    models[name]=_unavailable_model(exc)
                    model_cache.pop(name,None)
                    if name=='sabr':fit=None
                    issues.append(f'{name.upper()} indisponible : {exc}')
            if any(v[2] for v in target_values):warnings.append('Ancien snapshot : dates libres calculées par interpolation du maillage affiché. Recalculer les courbes normales pour utiliser les polynômes exacts.')
            parameters=None if fit is None else {key:fit[key] for key in ('alpha','beta','rho','nu')}
            rows.append(dict(start_date=str(ql_to_date(start)),end_date=str(ql_to_date(end)),fixing_date=fixing_string,payment_date=payment_string,forward_rate=F,expiry_years=T,accrual_fraction=accrual,discount_factor=df,targets_interpolated=any(v[1] for v in target_values),models=models,parameters=parameters,issues=issues))
            cached.append(dict(forward=F,expiry=T,annuity=A,models=model_cache))
        total_annuity=fsum(c['annuity'] for c in cached);atm_rate=fsum(c['forward']*c['annuity'] for c in cached)/total_annuity;totals={}
        for name in ('sabr','cubic'):
            missing=next((row for row in rows if row['models'][name]['status']!='ready'),None)
            if missing is not None:
                totals[name]=_unavailable_model(f"Caplet fixing {missing['fixing_date']} : {missing['models'][name]['reason']}",total=True)
                warnings.append(f"{name.upper()} indisponible : {totals[name]['reason']}")
                continue
            atm_vol=None;model_strikes=cached[0]['models'][name]['strikes']
            if model_strikes[0]<=atm_rate<=model_strikes[-1]:
                atm_price=0.
                for item in cached:
                    F,T,A=item['forward'],item['expiry'],item['annuity'];model=item['models'][name];fit=model['fit']
                    vol=sabr_normal_vol(F,atm_rate,T,fit['alpha'],fit['beta'],fit['rho'],fit['nu']) if name=='sabr' else normal_strike_vol(model['strikes'],model['targets'],atm_rate)
                    atm_price+=normal_price_vega(F,atm_rate,vol,T,A,1.)[0]
                atm_vol=brentq(lambda vol:fsum(normal_price_vega(c['forward'],atm_rate,vol,c['expiry'],c['annuity'],1.)[0] for c in cached)-atm_price,1e-10,1.,xtol=1e-14)
            else:warnings.append(f'ATM du cap hors du domaine observé {name.upper()} : volatilité flat ATM indisponible.')
            price=fsum(row['models'][name]['price'] for row in rows)
            totals[name]=dict(status='ready',reason=None,price=price,unit_price=price/req.notional,normal_vega_eur_bp=fsum(row['models'][name]['normal_vega_eur_bp'] for row in rows),atm_normal_vol=atm_vol)
    n_issues=sum(bool(row['issues']) for row in rows)
    if n_issues:warnings.append(f'{n_issues} caplet(s) avec modèle indisponible, paramètres aux bornes ou contrôle de forme à examiner dans le détail des flux.')
    warnings.append('Vega normale parallèle par +1 bp, courbes et smile figés ; aucune recalibration. ATM flat du cap obtenue par inversion Bachelier au strike moyen pondéré par les annuités.')
    return dict(valuation_date=str(req.valuation_date),source_token=req.source_token,index_tenor=req.index_tenor,strike=req.strike,notional=req.notional,atm_rate=atm_rate,models=totals,caplets=rows,warnings=list(dict.fromkeys(warnings)),calibration_strikes=dict(cubic=strikes.tolist(),sabr=positive_strikes.tolist()),conventions=dict(currency='EUR',discounting='ESTR',calendar='TARGET',business_day_convention=req.business_day_convention,day_count=req.day_count,date_generation=req.date_generation,fixing_days=req.fixing_days,payment_lag=req.payment_lag,roll_day=req.roll_day,end_of_month=req.end_of_month,time_method=req.time_method,weighting=req.weighting,atm_anchor=req.atm_anchor,beta=req.beta))
