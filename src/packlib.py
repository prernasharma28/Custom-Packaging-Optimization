"""packlib.py

Simple 3D bin-packing prototype (greedy / first-fit-decreasing by volume).

This is a heuristic prototype for grouping multiple products into standard
box sizes. It intentionally keeps the logic simple and explainable for a
research demo: it allows rotation (permutes product dimensions) and packs
items into the first standard box that can fit them by dimension and
also keeps volume accounting to avoid obviously exceeding capacity.

Limitations: this is not an optimal 3D bin-packing solver. For improved
results you can integrate OR-Tools or a specialized solver.
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict
import itertools


@dataclass
class BoxSpec:
    name: str
    inner_l: float
    inner_w: float
    inner_h: float

    @property
    def volume(self) -> float:
        return self.inner_l * self.inner_w * self.inner_h


@dataclass
class Product:
    id: int
    l: float
    w: float
    h: float

    @property
    def volume(self) -> float:
        return self.l * self.w * self.h


def _fits_by_dimensions(box: BoxSpec, product: Product) -> bool:
    # Allow rotation: check any permutation of product dims
    dims = (product.l, product.w, product.h)
    for perm in set(itertools.permutations(dims, 3)):
        if perm[0] <= box.inner_l + 1e-9 and perm[1] <= box.inner_w + 1e-9 and perm[2] <= box.inner_h + 1e-9:
            return True
    return False


def pack_products_to_standard_boxes(products: List[Product], standard_boxes: List[BoxSpec]) -> Dict[str, List[List[int]]]:
    """Greedy first-fit-decreasing by product volume.

    Returns mapping: box_name -> list of boxes (each box is list of product ids)
    """
    # Sort products by decreasing volume
    prods = sorted(products, key=lambda p: p.volume, reverse=True)

    # For each standard box size, we will maintain a list of current boxes with their used volume and product ids
    packing: Dict[str, List[Tuple[float, List[int]]]] = {b.name: [] for b in standard_boxes}

    for p in prods:
        placed = False
        # Try each standard box (smallest first to avoid waste)
        for box in sorted(standard_boxes, key=lambda b: b.volume):
            if not _fits_by_dimensions(box, p):
                continue
            # Try to fit into an existing open box (simple volume-check)
            for idx, (used_vol, ids) in enumerate(packing[box.name]):
                if used_vol + p.volume <= box.volume + 1e-9:
                    packing[box.name][idx] = (used_vol + p.volume, ids + [p.id])
                    placed = True
                    break
            if placed:
                break
            # Open a new box of this type and place product
            packing[box.name].append((p.volume, [p.id]))
            placed = True
            break
        if not placed:
            # Product could not fit into any standard box by dimension; mark under 'unfit'
            packing.setdefault('unfit', []).append((p.volume, [p.id]))

    # Convert to simpler mapping: box_name -> list of boxes (product id lists)
    result: Dict[str, List[List[int]]] = {}
    for name, boxes in packing.items():
        result[name] = [ids for (_vol, ids) in boxes]
    return result


def example_standard_boxes() -> List[BoxSpec]:
    # Common sample sizes (inner dims in cm) - adjust to your business rules
    return [
        BoxSpec('small', 20.0, 15.0, 10.0),
        BoxSpec('medium', 40.0, 30.0, 20.0),
        BoxSpec('large', 60.0, 40.0, 40.0),
    ]


if __name__ == '__main__':
    # tiny demo
    items = [Product(i, *dims) for i, dims in enumerate([(10,5,2), (15,10,3), (30,20,5), (5,5,5), (50,30,10)])]
    boxes = example_standard_boxes()
    out = pack_products_to_standard_boxes(items, boxes)
    print('Packing result:', out)
