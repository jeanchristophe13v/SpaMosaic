import torch
import torch.nn.functional as F
from spamosaic.loss import CL_loss,BPR_loss,PairedAlignmentLoss
from spamosaic.topola import full_topola_distance,reference_topola_features

torch.manual_seed(7)
for n in (2,5):
    x=torch.randn(2*n,2*n,dtype=torch.float64,requires_grad=True)
    s=(x+x.T)/2
    for new,old in ((PairedAlignmentLoss(n,"cl","all"),CL_loss(n,2)),
                    (PairedAlignmentLoss(n,"bpr","cross"),BPR_loss(n,2))):
        torch.testing.assert_close(new(s),old(s))
        torch.testing.assert_close(torch.autograd.grad(new(s),x,retain_graph=True)[0],
                                   torch.autograd.grad(old(s),x,retain_graph=True)[0])
    for objective in ("cl","bpr"):
        for negatives in ("all","cross"):
            terms=[]
            for i in range(2*n):
                p=(i+n)%(2*n)
                js=[j for j in range(2*n) if j!=i and j!=p and (negatives=="all" or i//n!=j//n)]
                if objective=="bpr":
                    terms.extend(F.softplus(s[i,j]-s[i,p]) for j in js)
                else:
                    terms.append(torch.logsumexp(s[i,js+[p]],0)-s[i,p])
            torch.testing.assert_close(PairedAlignmentLoss(n,objective,negatives)(s),torch.stack(terms).mean())
for lam in (1e-7,1e-3,10.,1000.):
    e=torch.randn(14,4,dtype=torch.float64,requires_grad=True)
    ids=torch.tensor([0,2,5,7,9,12])
    f=reference_topola_features(e[ids],e,lam)
    actual=f@f.T
    expected=full_topola_distance(e,lam)[ids][:,ids]
    torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-8)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(),e,retain_graph=True)[0],
                               torch.autograd.grad(expected.square().sum(),e)[0],atol=1e-10,rtol=1e-8)
q=torch.randn(4,3,dtype=torch.float64,requires_grad=True)
r=torch.randn(9,3,dtype=torch.float64,requires_grad=True)
torch.autograd.gradcheck(lambda q,r:reference_topola_features(q,r,.001),(q,r))
print("PASS: CL/BPR legacy values+gradients; all four masks vs enumeration; reference kernel values+gradients; independent query/reference gradcheck")
