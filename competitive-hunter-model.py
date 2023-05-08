# Databricks notebook source
# MAGIC %md
# MAGIC # Competive Hunter Model - Tigers and Dinosaurs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Abstract
# MAGIC 
# MAGIC Tigers and dinosaurs are generally “competitive species” (as opposed to predator-prey). 
# MAGIC Although on rare occasions they may prey on each other (especially trying to gang up on the newborn), for the most part they share the same territory and often fight over prey.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Introduction
# MAGIC 
# MAGIC Changes in population of Tiger and Dinosaur are depicted as below: 
# MAGIC 
# MAGIC $$\frac {dT}{dt} = k_1T - k_3TD$$
# MAGIC 
# MAGIC $$\frac {dD}{dt} = k_2D - k_4TD$$
# MAGIC 
# MAGIC where 
# MAGIC * T: population of Tiger
# MAGIC * D: population of Dinosaur
# MAGIC * k1: birth rate of Tiger
# MAGIC * k2: birth rate of Dinosaur
# MAGIC * k3: death rate of Tiger
# MAGIC * k4: death rate of Dinosaur
# MAGIC 
# MAGIC #### 1. Derive equations for the equilibrium values of the two populations as a function of the four constants​. 
# MAGIC 
# MAGIC At equilibrium: T = Tn, D = Dn. That means the pouplation of tigers and dinosaurs are constant over time, or dT/dt = 0 and dD/dt = 0
# MAGIC 
# MAGIC $$\frac {dT}{dt} = k_1T - k_3TD = 0 $$
# MAGIC 
# MAGIC $$\frac {dD}{dt} = k_2D - k_4TD = 0 $$
# MAGIC 
# MAGIC Then, we have:
# MAGIC 
# MAGIC $$T = \frac{k_2}{k_4}$$
# MAGIC 
# MAGIC $$D = \frac{k_1}{k_3}$$
# MAGIC 
# MAGIC 
# MAGIC #### 2. Derive closed-form, analytical solutions for the two differential equations under the assumption that the death rates are zero​ 
# MAGIC 
# MAGIC If we assume that the death rates of both species are zero, then the differential equations become:
# MAGIC 
# MAGIC $$\frac {dT}{dt} = k_1T$$
# MAGIC 
# MAGIC $$\frac {dD}{dt} = k_2D$$
# MAGIC 
# MAGIC ##### a. For tiger population:
# MAGIC 
# MAGIC dT/dt = k1*T
# MAGIC 
# MAGIC Separation of varables
# MAGIC 
# MAGIC dT/T = k1*dt
# MAGIC 
# MAGIC Integrate both sides:
# MAGIC 
# MAGIC \\(ln(T) = k1*t + C\\)
# MAGIC where C is an arbitrary constant of integration exponentiate both sides:
# MAGIC 
# MAGIC \\(T = e^{(k1*t+C)}\\)
# MAGIC 
# MAGIC Given T0 initial population of Tiger at time t0, we can find C as:
# MAGIC 
# MAGIC \\(T_0 = Ce^{k_1*t_0}\\)
# MAGIC 
# MAGIC \\(C = T_0/e^{k_1*t_0}\\)
# MAGIC 
# MAGIC Assuming that t_0 = 0 for timestep 0, we have: \\(C = T_0\\)
# MAGIC 
# MAGIC Therefore, we have an exponential solution for T based on T0, k1, t:
# MAGIC 
# MAGIC \\(T_{anl} = T_0e^{k_1t_0}\\) where T0 is the initial population of Tigers
# MAGIC 
# MAGIC ##### b. For dinosaur population:
# MAGIC 
# MAGIC \\(dD/dt = k2*D\\)
# MAGIC 
# MAGIC Separation of varables
# MAGIC 
# MAGIC \\(dD/D = k2*dt\\)
# MAGIC 
# MAGIC Integrate both sides:
# MAGIC 
# MAGIC \\(ln(D) = k2*t + C\\), where C is an arbitrary constant of integration exponentiate both sides
# MAGIC 
# MAGIC Using similar logic to estimate population of Tiger above, we have:
# MAGIC 
# MAGIC \\(D = e^{(k2*t+C)}\\)
# MAGIC 
# MAGIC \\(D_{anl} = D_0e^{(k_2t)}\\) where D0 is the initial population of Dinosaurs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify
# MAGIC 
# MAGIC ### The first step: 
# MAGIC assume that death rates are zero, verify model using python and Insight Maker

# COMMAND ----------

k1 = 0.022
k2 = 0.04
k3 = 0.001
k4 = 0.0055
T0 = 22
D0 = 78
max_time = 48

# COMMAND ----------

import numpy as np

def sol(initial_pop, birth_rate, death_rate, max_time):
  T = np.arange(0, max_time+1)
  population = np.zeros(max_time+1)
  population[0] = initial_pop
  for t in range(1, max_time+1):
      population[t] = initial_pop * np.exp(birth_rate * t)

  return T, population


# COMMAND ----------

time, tigers = sol(initial_pop=T0, birth_rate=k1, death_rate=0, max_time=max_time)

# COMMAND ----------

time, dinosaurs = sol(initial_pop=D0, birth_rate=k2, death_rate=0, max_time=max_time)

# COMMAND ----------

import pandas as pd
df = pd.DataFrame({'t': time, 'tigers': tigers, 'dinosaurs': dinosaurs})
display(df)

# COMMAND ----------

display(dbutils.fs.mounts())

# COMMAND ----------

# MAGIC %md
# MAGIC Based on, InsightMaker Chart for tigers and dinosaurs population with 0 death rate:
# MAGIC At time 48, we have
# MAGIC * Tigers: 512
# MAGIC * Dinosaurs: 62
# MAGIC 
# MAGIC However, based on the analytics solution, we have:
# MAGIC * Tigers: 63
# MAGIC * Dinosaurs: 532
# MAGIC 
# MAGIC The difference is most likely due to different methodologies. InsightMaker uses Euler method to iteratively calculate the populations at each timestep. While the analytics solution use closed-form, integration function.  
# MAGIC 
# MAGIC <img src="https://github.com/anhhchu/databricks-demo/blob/master/simulation/nodeath.png?raw=true" style="width:50 height:50">

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Verify 2: 
# MAGIC 
# MAGIC Verify the equilibrium of 2 populations based on these equations:
# MAGIC 
# MAGIC $$T_{eq} = \frac{k2}{k4}$$
# MAGIC 
# MAGIC $$D_{eq} = \frac{k1}{k3}$$

# COMMAND ----------

tiger_eq = k2/k4
dinosaur_eq = k1/k3

print("Equilibrium of tigers is: ", tiger_eq)
print("Equilibrium of dinosaurs is:", dinosaur_eq) 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC InsightMaker chart of equilibrium value for Tiger population at 7.27 and Dinosaurs population at 22
# MAGIC 
# MAGIC ![equilibrium](https://github.com/anhhchu/databricks-demo/blob/master/simulation/equilibrium.png?raw=true)

# COMMAND ----------

# MAGIC %md
# MAGIC InsightMaker model of competitive hunter model
# MAGIC 
# MAGIC ![competitive_model](https://raw.githubusercontent.com/anhhchu/databricks-demo/48d097d604863808a500b9b10a215a915741c2d8/simulation/model.svg)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Reference
# MAGIC 
# MAGIC https://www.mathsisfun.com/calculus/separation-variables.html
# MAGIC 
# MAGIC https://www.mathsisfun.com/calculus/integration-introduction.html
