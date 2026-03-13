import matplotlib.pyplot as plt
import matplotlib.axes


class _Line:
    def __init__(self, axes: matplotlib.axes.Axes, name: str):
        (self.line,) = axes.plot([], [], label=name)
        self.set_data = lambda line, x, y: line.set_data(x, y)

        self.xdata = []
        self.ydata = []

    def Add(self, x, y):
        self.xdata.append(x)
        self.ydata.append(y)
        self.set_data(self.line, self.xdata, self.ydata)


class _Subplot:
    def __init__(self, axes: matplotlib.axes.Axes, title: str, value_names: list[str]):
        self.axes = axes
        self.axes.set_title(title)
        self.lines: dict[str, _Line] = {}
        for name in value_names:
            self.lines[name] = _Line(axes, name)

        axes.legend(loc="best")
        self.last_update_x = -1

    def Update(self, x: int, values: dict[str, float]):
        self.last_update_x = x
        for name, line in self.lines.items():
            if name in values:
                line.Add(x, values[name])

    def AutoScale(self, x: int):
        if self.last_update_x == x:
            self.axes.relim()
            self.axes.autoscale_view()


class Dashboard:
    # @example Dashboard("My Dashboard",{"Loss":["loss"],"Accuracy":["train_accuracy","test_accuracy"]})
    def __init__(
        self,
        title: str,
        value_sets: dict[str, list[str]],
        figsize: tuple[float, float] = plt.rcParams["figure.figsize"],
    ):
        plt.ion()
        self.fig = plt.figure(figsize=figsize)
        self.fig.suptitle(title)
        self.subplots: list[_Subplot] = []
        for i, (name, value_names) in enumerate(value_sets.items()):
            plot = self.fig.add_subplot(1, len(value_sets), i + 1)
            self.subplots.append(_Subplot(plot, name, value_names))
        self.xvalue = 0

    def Update(self, values: dict[str, float]):
        self.xvalue += 1
        for subplot in self.subplots:
            subplot.Update(self.xvalue, values)
        for subplot in self.subplots:
            subplot.AutoScale(self.xvalue)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
