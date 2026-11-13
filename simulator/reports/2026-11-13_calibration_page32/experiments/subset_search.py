"""Exact task-weighted selection with explicitly prioritized target ranks."""
from fractions import Fraction
import math

import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp

TARGET=(32,.25)
SECONDARY=((16,.25),(8,.25))


def optimize(matrix,size):
    assert len(matrix)>=size and not matrix.isna().any().any()
    columns=list(matrix.columns);n=len(matrix);k=len(columns)
    scale=math.lcm(*(Fraction(float(v)).limit_denominator(10000).denominator for v in matrix.to_numpy().ravel()))
    assert scale<=100000
    values=np.rint(matrix.to_numpy()*scale).astype(np.int64)
    assert np.allclose(values/scale,matrix.to_numpy(),atol=1e-10)
    target=values[:,columns.index(TARGET)]
    difference=target[:,None]-np.delete(values,columns.index(TARGET),axis=1)
    constraints=np.vstack([np.column_stack((difference.T,-np.ones(k-1))),np.r_[np.ones(n),0]])
    first=milp(np.r_[np.zeros(n),-1],integrality=np.ones(n+1),
        bounds=Bounds(np.r_[np.zeros(n),-size*scale],np.r_[np.ones(n),size*scale]),
        constraints=LinearConstraint(constraints,np.r_[np.zeros(k-1),size],np.r_[np.full(k-1,np.inf),size]),
        options={'time_limit':300,'mip_rel_gap':0})
    if first.status!=0:return dict(outcome='Primary optimization incomplete',solver_status=int(first.status),message=first.message)
    margin=int(round(first.x[-1]));m=size*scale+1
    indicators=2*(k-1);width=n+indicators+1;last=width-1
    rows=[];lower=[];upper=[]
    def constrain(coeff,low,high):rows.append(coeff);lower.append(low);upper.append(high)
    for d in difference.T:
        row=np.zeros(width);row[:n]=d;constrain(row,margin,np.inf)
    row=np.zeros(width);row[:n]=1;constrain(row,size,size)
    for group,config in enumerate(SECONDARY):
        s=columns.index(config);rivals=[j for j in range(k) if j!=s]
        offset=n+group*(k-1)
        for j,rival in enumerate(rivals):
            row=np.zeros(width);row[:n]=values[:,rival]-values[:,s];row[offset+j]=-m
            # b=1 iff the rival exceeds this secondary configuration by at
            # least one exact score unit. Ties do not increase the rank.
            constrain(row,1-m,0)
        row=np.zeros(width);row[offset:offset+k-1]=1;row[last]=-1
        constrain(row,-np.inf,-1)
    lb=np.zeros(width);ub=np.ones(width);lb[last]=1;ub[last]=k
    objective=np.zeros(width);objective[n:last]=1;objective[last]=indicators+1
    second=milp(objective,integrality=np.ones(width),bounds=Bounds(lb,ub),
        constraints=LinearConstraint(np.asarray(rows),lower,upper),options={'time_limit':300,'mip_rel_gap':0})
    if second.status!=0:return dict(outcome='Secondary-rank optimization incomplete',primary_margin_units=margin,
        score_scale=scale,solver_status=int(second.status),message=second.message)
    worst_rank=int(round(second.x[last]));indicator_sum=int(round(second.x[n:last].sum()))
    row=np.zeros(width);row[last]=1;constrain(row,worst_rank,worst_rank)
    row=np.zeros(width);row[n:last]=1;constrain(row,indicator_sum,indicator_sum)
    objective=np.zeros(width);objective[:n]=-target
    third=milp(objective,integrality=np.ones(width),bounds=Bounds(lb,ub),
        constraints=LinearConstraint(np.asarray(rows),lower,upper),options={'time_limit':300,'mip_rel_gap':0})
    if third.status!=0:return dict(outcome='Final tie-break optimization incomplete',solver_status=int(third.status),message=third.message)
    assert np.allclose(third.x,np.round(third.x),atol=1e-6)
    chosen=third.x[:n]>.5;assert chosen.sum()==size
    totals=values[chosen].sum(0)
    ranks={c:1+int((totals>totals[columns.index(c)]).sum()) for c in (TARGET,*SECONDARY)}
    actual_margin=int((totals[columns.index(TARGET)]-np.delete(totals,columns.index(TARGET))).min())
    assert actual_margin==margin
    assert max(ranks[c] for c in SECONDARY)==worst_rank
    assert sum(ranks[c] for c in SECONDARY)==indicator_sum+2
    return dict(optimal=True,task_ids=list(matrix.index[chosen]),subset_tasks=size,eligible_pool_tasks=n,
        score_scale=scale,minimum_margin_units=margin,minimum_margin_fraction=margin/(scale*size),
        target_score=float(totals[columns.index(TARGET)]/(scale*size)),
        target_rank=ranks[TARGET],secondary_ranks=[dict(A=c[0],alpha=c[1],rank=ranks[c],
            score=float(totals[columns.index(c)]/(scale*size))) for c in SECONDARY],
        outcome='unique first' if margin>0 else 'tied first' if margin==0 else 'cannot rank first in a subset of this size',
        stages_optimal=[True,True,True],objective_order=['maximum primary worst-rival margin',
            'minimum worst secondary rank then rank sum','maximum primary score'])
