
# TODO

## Notes 2020 / 11 / 16

- Faire marcher MOM : la closure de la mom strategy foire car je pense que le tableau des grads blocks est en lecture seule
- Faire marcher MOM : verifier que ca match quand block_size=sample size
  

## Reprise 2020 / 11 / 13

- Base OK pour CGD avec logistic et L1 / L2

- Implementer comparaisons avec scikit-learn

- Faire marcher CGD pour des matrices sparses et mettre tests sur X et y
- Mettre weights dans loss et gerer les calculs dans ce cas
- Implementer la strategie "mom-mle" pour CGD 

A terme ce qui serait bien serait : "erm-mle", "mom-mle", "erm-smp", "mom-smp" 

- Mettre les lip_constants, lip_max, lip_mean dans les modeles
- property dans solver pour le step='best' par defaut
- tests pour les solvers...
