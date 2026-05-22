import time
import random
from collections import defaultdict
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.times = defaultdict(list)

    @contextmanager
    def measure(self, category):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.times[category].append(elapsed)

    def get_times(self, category):
        return self.times[category] if category in self.times else self.times

    def get_stats(self, category):
        if category not in self.times or not self.times[category]:
            return {"count": 0, "total": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}

        durations = self.times[category]
        return {
            "count": len(durations),
            "total": sum(durations),
            "mean": sum(durations)/len(durations),
            "min": min(durations),
            "max": max(durations),
        }

    def reset(self):
        self.times.clear()
        # for category in self.times:
        #     self.times[category] = [0]

    def summary(self, categories=None):
        if categories is None:
            categories = self.times.keys()

        summary = {}

        for c in categories:
            stats = self.get_stats(c)
            if stats.get("count") > 0:
                summary[c] = {**stats}

        return summary


class TimeNode:
    __slots__ = ("category", "tag", "parent", "children", "times")

    def __init__(self, category, parent):
        self.category = category
        self.parent = parent
        self.children = {}              # category -> Node
        self.times = defaultdict(list)  # tag -> times


class TimeTree:
    def __init__(self, print_func=None):
        self.root = TimeNode(None, None)
        self.stack = [self.root]
        self.print_func = print_func or print

    @contextmanager
    def measure(self, category):
        parent = self.stack[-1]
        if category not in parent.children:
            parent.children[category] = TimeNode(category, parent)
        node = parent.children[category]
        self.stack.append(node)
        measurement = {"tag": None}
        start = time.perf_counter()
        try:
            yield measurement
        finally:
            elapsed = time.perf_counter() - start
            self.stack.pop()
            node.times[measurement["tag"]].append(elapsed)

    def reset(self, category=None):
        # TODO: implement resetting specific category
        if not category:
            self.root = TimeNode(None, None)
            self.stack = [self.root]

    def to_dict(self):
        result = {}

        def dfs(node, path):
            if node.category is not None:
                key = tuple(path + [node.category])
                result[key] = dict(node.times)

            for child in node.children.values():
                dfs(child, path + ([node.category] if node.category else []))

        dfs(self.root, [])
        return result

    def summary(self):
        result = {}
        for path, times in self.to_dict().items():
            result[path] = {}
            for tag, values in times.items():
                if values:
                    result[path][tag] = {
                        "count": len(values),
                        "total": sum(values),
                        "avg": sum(values)/len(values),
                        "min": min(values),
                        "max": max(values),
                    }
        return result

    # def print_tree(self):
    #     def fmt(times):
    #         t_count = len(times)
    #         t_total = sum(times)
    #         t_avg = t_total / t_count if t_count else 0
    #         t_min = min(times) if times else 0
    #         t_max = max(times) if times else 0
    #         return f"(x{t_count}) total={t_total*1000:.3f}ms avg={t_avg*1000:.3f}ms min={t_min*1000:.3f}ms max={t_max*1000:.3f}ms"

    #     def dfs(node, prefix="", is_last=True):
    #         if node.category is not None:
    #             connector = ("└─" if is_last else "├─") if node.parent is not self.root else ""
    #             # connector = "  " if node.parent is not self.root else ""
    #             line_prefix = prefix + connector

    #             tags = list(node.times.keys())

    #             if len(tags) == 1 and tags[0] is None:
    #                 self.print_func(f"{line_prefix}■ {node.category} {fmt(node.times[None])}")

    #             else:
    #                 self.print_func(f"{line_prefix}□ {node.category}")
    #                 # self.print_func(f"{line_prefix}▼ {node.category}")
    #                 ordered_tags = sorted(tags, key=lambda x: (x is None, x))
    #                 for i, tag in enumerate(ordered_tags):
    #                     tag_last = (i == len(ordered_tags) - 1)
    #                     tag_connector = "└─" if tag_last else "├─"
    #                     # tag_connector = "  "
    #                     tag_prefix = prefix + ("  " if is_last else "│ ")
    #                     # tag_prefix = prefix + "  "
    #                     self.print_func(f"{tag_prefix}{tag_connector}[{tag}] {fmt(node.times[tag])}")

    #         children = list(node.children.values())
    #         n = len(children)

    #         for i, child in enumerate(children):
    #             child_last = (i == n - 1)
    #             new_prefix = prefix + (("  " if is_last else "│ ") if (node is not self.root and node.parent is not self.root) else "")
    #             # new_prefix = prefix + ("  " if (node is not self.root and node.parent is not self.root) else "")
    #             dfs(child, new_prefix, child_last)

    #     dfs(self.root)

    def print_tree(self):
        def fmt(times):
            t_count = len(times)
            t_total = sum(times)
            t_avg = t_total / t_count if t_count else 0
            t_min = min(times) if times else 0
            t_max = max(times) if times else 0
            return f"({t_count}) total={t_total:.6f}s avg={t_avg:.6f}s min={t_min:.6f}s max={t_max:.6f}s"

        def dfs(node, prefix=""):
            if node.category is not None:
                tags = list(node.times.keys())

                if len(tags) == 1 and tags[0] is None:
                    self.print_func(f"{prefix}■ {node.category} {fmt(node.times[None])}")

                else:
                    self.print_func(f"{prefix}□ {node.category}")  # category name already known

                    ordered_tags = sorted(tags, key=lambda x: (x is None, x))
                    for i, tag in enumerate(ordered_tags):
                        tag_connector = "└─" if i == len(ordered_tags) - 1 else "├─"
                        self.print_func(f"{prefix}{tag_connector}[{tag}] {fmt(node.times[tag])}")

            children = list(node.children.values())

            for child in children:
                dfs(child, prefix + "│ " if node is not self.root else "")

            if node.category is not None:
                tags = list(node.times.keys())
                if len(tags) == 1 and tags[0] is None and len(node.children) > 0:
                    self.print_func(f"{prefix}└")

        dfs(self.root)


if __name__ == "__main__":
    import pprint

    def recv_msg():
        children = [1, 2, 3, 4]
        time.sleep(0.01)
        return random.choice(children)

    timer = TimeTree()

    for _ in range(2):
        with timer.measure("outer"):
            for _ in range(2):
                with timer.measure("inner"):
                    with timer.measure("A"):
                        for _ in range(4):
                            with timer.measure("recv_msg") as t:
                                idx = recv_msg()
                                t["tag"] = idx
                    with timer.measure("B"):
                        for _ in range(4):
                            with timer.measure("recv_msg") as t:
                                idx = recv_msg()
                                t["tag"] = idx

    pprint.pp(timer.summary())
    timer.print_tree()
