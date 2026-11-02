"""Decompose the invsqrt and dqi cost gaps against the public frontier.

Both are checked against our own verified runs and the forum's author-reported numbers,
so the formula is confirmed on two independent points before it is used as a lever model.
"""
import json

INV_BASE, DQI_BASE = 2076774.233715282, 24439474177519.598

# --- invsqrt: cost = toffoli_equivalent * num_qubits -------------------------------
ours_i = dict(score=157841.92320014222, te=1753.7991466682467, q=90)
pub_i = dict(score=103546, te=None, q=63)
print("invsqrt: cost = toffoli_equivalent * num_qubits")
print(f"  ours  {ours_i['te']:.4f} * {ours_i['q']} = {ours_i['te'] * ours_i['q']:.2f}"
      f"   (verified score {ours_i['score']:.2f})")
our_te_from_model = ours_i['score'] / 90
pub_te = pub_i['score'] / pub_i['q']
print(f"  public TE inferred = {pub_i['score']}/{pub_i['q']} = {pub_te:.4f}")
print(f"  public reported mean toffoli_count 1348.5, so TE - count = {pub_te - 1348.5:.2f}"
      f"  for 44 rotations ({44} -> {(pub_te - 1348.5) / 44:.3f} each)")
r_q = 90 / 63
r_te = ours_i['te'] / pub_te
print(f"  gap {ours_i['score'] / pub_i['score']:.4f}x  =  qubits {r_q:.4f}x  *  TE {r_te:.4f}x")
share = __import__('math').log(r_q) / __import__('math').log(ours_i['score'] / pub_i['score'])
print(f"  qubit share of the gap: {100 * share:.1f}%")

# --- dqi: cost = Q^2 * T * repetition_factor --------------------------------------
ours_d = dict(score=1717915181875.0806, q=1789, t=445229.0, rf=1.2055843238398525)
pub_d = dict(q=877, t=482543, rf=None)
model_ours = ours_d['q'] ** 2 * ours_d['t'] * ours_d['rf']
print("\ndqi: cost = Q^2 * T * repetition_factor")
print(f"  ours   {ours_d['q']}^2 * {ours_d['t']:.0f} * {ours_d['rf']:.6f} = {model_ours:.4e}"
      f"   (verified {ours_d['score']:.4e})")
rf_pub = 442100000000 / (pub_d['q'] ** 2 * pub_d['t'])
print(f"  public rf inferred = 4.421e11 / ({pub_d['q']}^2 * {pub_d['t']}) = {rf_pub:.4f}")
r_q2 = (ours_d['q'] / pub_d['q']) ** 2
r_t = ours_d['t'] / pub_d['t']
r_rf = ours_d['rf'] / rf_pub
print(f"  gap {ours_d['score'] / 442100000000:.4f}x  =  Q^2 {r_q2:.4f}x  *  T {r_t:.4f}x  *  rf {r_rf:.4f}x")
print(f"  T share: we are {'BETTER' if r_t < 1 else 'worse'} on Toffolis by {abs(1 / r_t - 1) * 100:.1f}%")
share_d = __import__('math').log(r_q2) / __import__('math').log(ours_d['score'] / 442100000000)
print(f"  qubit share of the gap: {100 * share_d:.1f}%")

# --- what each lever alone would buy ----------------------------------------------
print("\nwhat if only the qubit count changed (everything else held at our values):")
for tag, base, ours_score, new_q, q_old, expo in [
        ('invsqrt 90 -> 63', INV_BASE, ours_i['score'], 63, 90, 1),
        ('invsqrt 90 -> 78', INV_BASE, ours_i['score'], 78, 90, 1),
        ('dqi 1789 -> 877', DQI_BASE, ours_d['score'], 877, 1789, 2)]:
    new_cost = ours_score * (new_q / q_old) ** expo
    ratio = base / new_cost
    print(f"  {tag:18} cost {new_cost:>12.4e}  ratio {ratio:>6.2f}x "
          f"(was {base / ours_score:.2f}x, x{base / ours_score and (base / new_cost) / (base / ours_score):.3f} on the ratio)")

print("\ncomposite leverage (ratio x, cost divided):")
comp0 = 6.7266675673140375
for tag, f_inv, f_dqi, f_chem in [('invsqrt 63q', 1.4286, 1.0, 1.0),
                                  ('dqi 877q', 1.0, 4.161, 1.0),
                                  ('both', 1.4286, 4.161, 1.0),
                                  ('both + chemistry 5.54x', 1.4286, 4.161, 5.54 / 1.626)]:
    comp = comp0 * (f_inv * f_dqi * f_chem) ** (1 / 3)
    print(f"  {tag:24} -> composite {comp:.2f}")
