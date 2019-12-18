from linlearn.solver import History
from time import sleep


history = History('Truc', 4, verbose=True)

history.update(n_iter=0, obj=1.2, tol=3.2, blabla=4, update_bar=False)
sleep(1)
history.update(n_iter=1, obj=1.2, tol=3.2, blabla=4)
sleep(1)
history.update(n_iter=2, obj=1.2, tol=3.2, blabla=4)
sleep(1)
history.update(n_iter=3, obj=1.2, tol=3.2, blabla=4)
sleep(1)
# history.update(n_iter=4, obj=1.2, tol=3.2, blabla=4)
# sleep(1)
# history.update(n_iter=4, obj=1.2, tol=3.2, blabla=4)
# sleep(1)

# history.update(n_iter=4, obj=1.2, tol=3.2, blabla=4)
# sleep(1)
# history.update(n_iter=5, obj=1.2, tol=3.2, blabla=4)
# sleep(1)
# history.update(n_iter=5, obj=1.2, tol=3.2, blabla=4)
# sleep(1)
# history.update(n_iter=5, obj=1.2, tol=3.2, blabla=4)
# sleep(1)
# history.update(n_iter=5, obj=1.2, tol=3.2, blabla=4)
# sleep(1)
# history.update(n_iter=5, obj=1.2, tol=3.2, blabla=4)
# sleep(2)

history.print()
