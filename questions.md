# Questions:

- Come estrapolare gli elementi di matrice: non capisco i conti fatti in KMBSM. Altrimenti come ho pensato io andrebbe bene?

- Le directories ./log, ./dir, ./data, etc che sono? Che ci dovrebbe essere dentro? Gli ensemble da dove li tiro fuori? Non posso di certo generarli io.

- Why mixed action and not both O(a) improved fermions or both maximally twisted mass?
	La risposta dovrebbe essere nell'articolo di Frezzotti-Rossi

- Unphysical zero modes of Wilson fermions and tmQCD: skill yourself

- A cosa serve il Wilson Flow? Per generare le configurazioni di Gauge giusto? Beh direi di sì, ma non sono proprio sicurissimo.

- Cosa cambia tra il calcolo di correlatori a n-punti con le conidizioni periodiche al contorno rispetto alle obc? A me sembra che il calcolo sia stato fatto sempre allo stesso modo, con la divverenza che le timeslices di contorno non si possono utilizzare. Devo chiarire questa cosa.

- Inoltre perché ad un certo punto spunta fuori il tempo T negli esponenziali dei correlatori?

- Match delle masse di valenza con quelle dei quark: è saltato fuori che, sebbene entrambe O(a)-improved, matchare direttamente le masse dei quark dà meno fluttuazioni statistiche. Forse converrebbe questo.

- Perché l'operatore O_1 ha bisogno della corrente assiale invece che della pseudoscalare? Vedi su KMBSM >> RISOLTO.

- Perché il fattore 8/3 di fronte all'elemento di matrice del modello standard? 

***

## About code:

- Dove sono i dati degli ensemble? Come faccio a sapere se li ho o meno? Come li do in pasto al programma?

- Con che formato viene fatto lo storage dei correlatori? Io vorrei fare un'analisi dati in python, perché è più semplice. 
