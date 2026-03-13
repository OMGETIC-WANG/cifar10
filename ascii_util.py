import typing


def _GetDigitCount(x: int) -> int:
    if x == 0:
        return 0
    cnt = 0
    while x:
        cnt += 1
        x //= 10
    return cnt


def _Clamp(x, min, max):
    if x < min:
        return min
    if x > max:
        return max
    return x


class ProgressBar:
    def __init__(self, title: str, total: int, bar_len=32, show_percent: bool = True) -> None:
        self.title = title
        self.total = total
        self._num_len = _GetDigitCount(total)
        self._bar_len = bar_len
        self.show_percent = show_percent
        self.Update(0)

    def Update(self, current: int, append_message: str = ""):
        line = "\r\x1b[K"
        line += f"{self.title} {current:>{self._num_len}}/{self.total} "
        if self.show_percent:
            line += f"({float(current)/self.total*100:>6.2f}%) "
        l = _Clamp(int(float(current) / self.total * self._bar_len), 0, self._bar_len)
        line += f"[{'#'*l}{' '*(self._bar_len-l)}] "
        line += append_message
        print(line, end="", flush=True)

    def End(self):
        print("", flush=True)


def _CastToStr(val):
    return val if isinstance(val, str) else str(val)


class _ColumnDesc:
    def __init__(self, name: str) -> None:
        self.name = name
        self.items: list[str] = []
        self.max_len = len(name)

    def __getitem__(self, index: int):
        return self.items[index]

    def AddItem(self, item: str):
        self.items.append(item)
        self.max_len = max(len(item), self.max_len)

    def RemoveItem(self, index: int):
        item = self.items[index]
        self.items.pop(index)
        if len(item) == self.max_len:
            self.max_len = len(self.name)
            for x in self.items:
                self.max_len = max(len(x), self.max_len)

    def AlignedName(self):
        return _LeftAlign(self.name, self.max_len)

    def AlignedSplitLine(self):
        return _LeftAlign("-" * len(self.name), self.max_len)

    def AlignedItem(self, index: int):
        if index < len(self.items):
            return _LeftAlign(self.items[index], self.max_len)
        return " " * self.max_len


def _LeftAlign(s: str, align: int, align_char: str = " "):
    return s + align_char * max(align - len(s), 0)


class Table:
    def __init__(self, columns: typing.Sequence, vertical_blank: int = 4) -> None:
        self.columns = [_ColumnDesc(_CastToStr(col)) for col in columns]
        self.vertical_blank = vertical_blank

    def __getitem__(self, index: int):
        return self.columns[index]

    def AddRow(self, row):
        valid_size = min(len(self.columns), len(row))
        for col, rowitem in zip(self.columns[:valid_size], row[:valid_size]):
            col.AddItem(_CastToStr(rowitem))

    def RemoveRow(self, index: int):
        for col in self.columns:
            col.RemoveItem(index)

    def Print(self):
        if len(self.columns) == 0:
            print("<Empty table>")
            return
        self._PrintColumnTitle()
        self._PrintHorizenSplitLine()
        if len(self.columns[0].items) == 0:
            print("<Empty>")
            return
        for i in range(len(self.columns[0].items)):
            self._PrintRow(i)

    def _PrintColumnTitle(self):
        print((" " * self.vertical_blank).join(map(lambda x: x.AlignedName(), self.columns)))

    def _PrintHorizenSplitLine(self):
        print((" " * self.vertical_blank).join(map(lambda x: x.AlignedSplitLine(), self.columns)))

    def _PrintRow(self, index: int):
        print((" " * self.vertical_blank).join(map(lambda x: x.AlignedItem(index), self.columns)))
