import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * x_data*-0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# Learns best fit is W: [0.1], b: [0.3]


print(sess.run(W), sess.run(b))

fig, ax = plt.subplots()
for color in ['tab:blue']:
    n = 100
    x, y = x_data, y_data
    scale = 100
    ax.scatter(x, y, c=color, s=scale, label=color,
               alpha=1, edgecolors='none')
ax.grid(True)

line=[]
for x in range(0, 2):
  line+=[(sess.run(W) * x + sess.run(b))]
plt.plot(line)
plt.show()