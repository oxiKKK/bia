Explanation of setup:

- `functions.py` defines the functions.
- `plot.py` defines the matplot impl that plots the functions.
- `cvX.py` the lesson-specific script.

How to run (after setupping venv & requirements):

# CV7

```
python cv8.py
```
![alt text](images/fireflies.png)


# CV6

```bash
python cv6.py
```

![alt text](images/soma.png)


# CV5

```bash
python cv5.py
```

![alt text](images/pso.png)


# CV4

```bash
python cv4.py --function Ackley
```

![alt text](images/ge.png)


# CV3

```bash
python cv3.py
```

![alt text](images/tsp.png)

# CV2

```bash
python cv2.py
```

For each learning algorithm should look like this (non-animated this time):

![alt text](images/annealing.png)

- L,M temperature is hardcoded per-function in `cv2.py::temperature_for_function`
- Limited to N iterations (hardcoded in code)
- After closing the window, the new algorithm should appear on all of the functions.
- Applied to all functions from previous lesson.

# CV1

```bash
python cv1.py
```

For each learning algorithm should look something like this:

![alt text](images/blind.png)
![alt text](images/hill.png)

- It is animated
- Limited to N iterations (hardcoded in code)
- After closing the window, the new algorithm should appear on all of the functions.

Algorithms implemented:
- Blind Search
- Hill Climb
- Tabu Search (I think this was not needed, but i did FF)
