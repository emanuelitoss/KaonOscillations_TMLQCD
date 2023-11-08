# KaonOscillations_tmLQCD
This repo contains the work of an MSc thesis in theoretical physics.
The work consists in the evaluation of matrix elements of neutral Kaons oscillations with insertion of a complete set of intermediate mixing local operators $\Theta_i(x)$. Such operators usually are expressed in SUSY basis - clearly referred to Supersymmetry:

$\begin{equation}
    \begin{aligned}
       & \Theta_1 = [\bar s^a \gamma_\mu (1+\gamma_5) d^a] \cdot [ \bar s^b \gamma_\mu (1+\gamma_5) d^b ] \\
       & \Theta_2 = [\bar s^a  (1+\gamma_5) d^a ] \cdot [ \bar s^b (1+\gamma_5) d^b ] \\
       & \Theta_3 = [\bar s^a  (1+\gamma_5) d^b ] \cdot [ \bar s^b (1+\gamma_5) d^a ] \\
       & \Theta_4 = [\bar s^a  (1+\gamma_5) d^a ] \cdot [ \bar s^b (1-\gamma_5) d^b ] \\
       & \Theta_5 = [\bar s^a  (1+\gamma_5) d^b ] \cdot [ \bar s^b (1-\gamma_5) d^a ] \\
       & \tilde\Theta_1 = [\bar s^a \gamma_\mu (1-\gamma_5) d^a] \cdot [ \bar s^b \gamma_\mu (1-\gamma_5) d^b ] \\
       & \tilde\Theta_2 = [\bar s^a  (1-\gamma_5) d^a] \cdot [ \bar s^b (1-\gamma_5) d^b ] \\
       & \tilde\Theta_3 = [\bar s^a  (1-\gamma_5) d^b] \cdot [ \bar s^b (1-\gamma_5) d^a ]
    \end{aligned}
\end{equation}$

The quantities that need to be evaluated are the renormalized matrix elements:
$$ \langle \bar K^0 | \hat\Theta_i^\text{ren} (\mu) | K^0 \rangle $$
but I will only evaluate the bare matrix elements.

The working lab is the `lattice QCD`. I work with Osterwalder Seiler maximally twisted valence quarks, Luscher-Weisz Guage fields and $N_F = 2+1$ improved sea quarks. $O(a)-$ improvement is achieved in all the elements (regularizations, renormalization constants, matrix elements, etc).
The 'new tool' consists in the implementation of <span style="color:orange">Open Boundary Conditions</span> (OBC) in time direction. They allow the simulation to run faster and to be optimized in a better way; on the contrary, time translation invariance achieved through periodic boundary conditions is broken.

---
---

# Repo organization
The simulation codes are into <span style="color:orange">tm-mesons-obc</span> folder. Inside it you can find other README file(s) and documents about the code. The path to the main program is:
`meson-correlators > mesons-master > correlators.c`.
The entire code is based on <span style="color:orange">openQCD-1.2</span> by S. Luescher: https://luscher.web.cern.ch/luscher/openQCD/index.html
Data analysis codes are written in Python language and lies in <span style="color:orange">data-analysis-kkbar</span>.

---

### Contacts:
rosiemanuele99@gmail.com, rosi.1812180@studenti.uniroma1.it

![Kaons are mixing!](kaon.jpg "Kaons oscillatoions")
