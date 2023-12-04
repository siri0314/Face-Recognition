import matplotlib.pyplot as plt


x_values1=[0,0.39817,1]
y_values1=[0,0.595339,1]


x_values2=[0,0.0888304,1]
y_values2=[0,0.222458,1]


x_values3=[0,0.00699953,1]
y_values3=[0,0.0105932,1]


#make axes
fig=plt.figure()


ax=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)
ax3=fig.add_subplot(111, label="3", frame_on=False)

ax.plot(x_values1, y_values1, color="C0")
ax.set_xlabel("tpr", color="C0")
ax.set_ylabel("fpr", color="C0")
ax.tick_params(axis='x', colors="C0")
ax.tick_params(axis='y', colors="C0")


ax.plot(x_values2, y_values2, color="C1")
ax.tick_params(axis='x', colors="C1")
ax.tick_params(axis='y', colors="C1")

ax.plot(x_values3, y_values3, color="C2")
ax.tick_params(axis='x', colors="C2")
ax.tick_params(axis='y', colors="C2")

plt.show()
