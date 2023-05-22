from math import sqrt, tan, pi
from pykeen.nn import Interaction, Embedding
from base.erc_model import ERCModel
from torch import FloatTensor, rand_like, min, max, where, logical_and, ones_like, zeros_like
from torch.nn import Module
from torch.nn.init import uniform_

tanpi8 = tan(pi/8)
tan3pi8 = tan(3*pi/8)
sqrt2 = sqrt(2)

def init_center(u):
    return rand_like(u) * 2 - 1

def init_margin(du):
    return ones_like(du) * 0.1

def boundaries(r):
    """
    Transform input (center, delta) embeddings as (min, max) boundaries
    """
    x, y, u, v, dx, dy, du, dv = r

    return (x - dx, x + dx, y - dy, y + dy, u - du, u + du, v - dv, v + dv)

def tighten(r):
    """
    Tighten input octagon
    """
    x, y, u, v, dx, dy, du, dv = r

    xmin = x - dx
    xmax = x + dx
    ymin = y - dy
    ymax = y + dy
    umin = u - du
    umax = u + du
    vmin = v - dv
    vmax = v + dv

    return max(xmin, max(vmin - ymax, max(ymin - umax, (vmin - umax) / 2))), \
        min(xmax, min(vmax - ymin, min(ymax - umin, (vmax - umin) / 2))), \
        max(ymin, max(umin + xmin, max(vmin - xmax, (umin + vmin) / 2))), \
        min(ymax, min(umax + xmax, min(vmax - xmin, (umax + vmax) / 2))), \
        max(umin, max(ymin - xmax, max(vmin - 2 * xmax, 2 * ymin - vmax))), \
        min(umax, min(ymax - xmin, min(vmax - 2 * xmin, 2 * ymax - vmin))), \
        max(vmin, max(xmin + ymin, max(umin + 2 * xmin, 2 * ymin - umax))), \
        min(vmax, min(xmax + ymax, min(umax + 2 * xmax, 2 * ymax - umin)))

def all3(t1, t2, t3):
    # TODO isn't multiple input already supported in PyTorch?
    return logical_and(t1, logical_and(t2, t3))

def is_inside(x, r, y):
    xmin, xmax, ymin, ymax, umin, umax, vmin, vmax = r

    u = y - x
    v = y + x

    return (xmin < x) & (x < xmax) \
        & (ymin < y) & (y < ymax) \
        & (umin < u) & (u < umax) \
        & (vmin < v) & (v < vmax)

def inside_dist_score(x, r, y):
    xmin, xmax, ymin, ymax, umin, umax, vmin, vmax = r

    # TODO the easy-to-compute score here isn't the same as taking the distance to the centroid of the octagon
    # to make computation easier: use 2 more parameter: center_octagon (x, y)

    return -min(x - xmin,
        min(xmax - x,
        min(y - ymin,
        min(ymax - y,
        min(y - x - umin,
        min(umax - y + x,
        min(y + x - vmin,
        vmax - y - x)))))))

def outside_dist_score(x, r, y):
    xmin, xmax, ymin, ymax, umin, umax, vmin, vmax = r

    # parameters of delimiter lines, s.t. y = var[0]*x + var[1]
    a = [tanpi8, vmin - xmin - tanpi8 * xmin]
    b = [-tanpi8, umax + xmax + tanpi8 * xmin]
    c = [-tan3pi8, ymax + tan3pi8 * (ymax - umax)]
    d = [tan3pi8, ymax - tan3pi8 * (vmax - ymax)]
    e = [tanpi8, vmax - xmax - tanpi8 * xmax]
    f = [-tanpi8, umin + xmax +tanpi8 * xmax]
    g = [-tan3pi8, umin + tan3pi8 * (ymin - umin)]
    h = [tan3pi8, ymin - tan3pi8 * (vmin - ymin)]

    return where(
        (x < xmin) & (y > a[0] * x + a[1]) & (y < b[0] * x + b[1]), # AB
        -(x - xmin),
    where(
        (y - x > umax) & (y > b[0] * x + b[1]) & (y < c[0] * x + c[1]), # BC
        (y - x - umax) / sqrt2,
    where(
        (y > ymax) & (y > c[0] * x + c[1]) & (y > d[0] * x + d[1]), # CD
        (y - ymax),
    where(
        (y + x > vmax) & (y < d[0] * x + d[1]) & (y > e[0] * x + e[1]), # DE
        (y + x - vmax) / sqrt2,
    where(
        (x > xmax) & (y < e[0] * x + e[1]) & (y > f[0] * x + f[1]), # EF
        (x - xmax),
    where(
        (y - x < umin) & (y < f[0] * x + f[1]) & (y > g[0] * x + g[1]), # FG
        -(y - x - umin) / sqrt2,
    where(
        (y < ymin) & (y < g[0] * x + g[1]) & (y < h[0] * x + h[1]), # GH
        -(y - ymin),
    where(
        (y + x < vmin) & (y > h[0] * x + h[1]) & (y < a[0] * x + a[1]), # HA
        -(y + x - vmin) / sqrt2,
        0)))))))) # exactly on the octagon's border

class OctagonsTripleInteraction(Interaction):

    entity_shape = ("d")
    relation_shape = ("d","d","d","d","d","d","d","d")

    def forward(self, h, r, t) -> FloatTensor:
        xmin, xmax, ymin, ymax, umin, umax, vmin, vmax = tighten(r)

        dx = (xmax - xmin) / 2
        dy = (ymax - ymin) / 2
        du = (umax - umin) / 2
        dv = (vmax - vmin) / 2
        
        length = min(dx, min(dy, min(du, dv))) + 0.0001

        #w = max(du, dv) # borrow idea from BoxE of making margin variable
        w = 1

        dist = w * (length + where(
            is_inside(h, r, t),
            inside_dist_score(h, r, t),
            outside_dist_score(h, r, t)
        ))
        
        return -dist.norm(dim=-1)

class OctagonsTypeInteraction(Interaction):

    entity_shape = ("d",)
    class_shape = ("d","d")

    def score_ec(self, i, c) -> FloatTensor:
        return self(i, c).unsqueeze(dim=-1)

    def forward(self, i, c) -> FloatTensor:
        x, dx = c

        # TODO include dx
        dist = i - x

        return -dist.norm(dim=-1)

class OctagonsModel(ERCModel):

    def __init__(self, *, embedding_dim: int = 50, **kwargs):
        super().__init__(
            triple_interaction=OctagonsTripleInteraction,
            type_interaction=OctagonsTypeInteraction,
            entity_representations=Embedding,
            entity_representations_kwargs=dict(
                embedding_dim=embedding_dim,
                initializer=uniform_,
                initializer_kwargs=dict(a=-1,b=1),
            ),
            relation_representations=Embedding,
            relation_representations_kwargs=[
                dict( # x
                    embedding_dim=embedding_dim,
                    initializer=zeros_like
                ),
                dict( # y
                    embedding_dim=embedding_dim,
                    initializer=zeros_like
                ),
                dict( # u
                    embedding_dim=embedding_dim,
                    initializer=init_center
                ),
                dict( # v
                    embedding_dim=embedding_dim,
                    initializer=zeros_like
                ),
                dict( # dx
                    embedding_dim=embedding_dim,
                    initializer=ones_like
                ),
                dict( # dy
                    embedding_dim=embedding_dim,
                    initializer=ones_like
                ),
                dict( # du
                    embedding_dim=embedding_dim,
                    # TODO random octagon around (x,y) instead
                    initializer=init_margin
                ),
                dict( # dv
                    embedding_dim=embedding_dim,
                    initializer=ones_like
                )
            ],
            class_representations=Embedding,
            class_representations_kwargs=[
                dict( # c
                    embedding_dim=embedding_dim,
                ),
                dict( # dc
                    embedding_dim=embedding_dim,
                )
            ],
            **kwargs
        )