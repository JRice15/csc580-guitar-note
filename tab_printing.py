import curses
import time
import math

def fretting_to_dict(fretting):
    """
    args:
        fretting: list of tuples, in (string_number, fret) format
    returns:
        dict mapping string name to fret
    """
    strings = " EADGBe"
    if not len(fretting):
        return {}
    dct = {strings[i]:fret for i,fret in fretting}
    return dct


class Tabulator():

    def __init__(self, maxlen=48):
        self.tabs = {x:"" for x in "EADGBe"}
        self.most_recent_fret = {x:None for x in "EADGBe"}
        self.maxlen = maxlen
        assert self.maxlen % 3 == 0
    
    def add_timestep(self, string_fret_map):
        """
        args:
            string_fret_map: dict, or list of tuples
        """
        if not isinstance(string_fret_map, dict):
            string_fret_map = fretting_to_dict(string_fret_map)
        for string,history in self.tabs.items():
            if string in string_fret_map:
                new_fret = string_fret_map[string]
                if self.most_recent_fret[string] != new_fret:
                    self.most_recent_fret[string] = new_fret
                    formatted = "{:->3}".format(string_fret_map[string])
                    self.tabs[string] += formatted
                else:
                    self.tabs[string] += "--~"
            else:
                self.most_recent_fret[string] = None
                self.tabs[string] += "---"


    def output_to_curses(self, screen):
        maxlines, maxlen = screen.getmaxyx()
        maxlen -= 3
        tablen = len(self.tabs["E"])
        steps = math.ceil(tablen / maxlen)
        total_lines = steps * 7
        overflow = max(total_lines - maxlines, 0)
        for step in range(steps):
            for i,string in enumerate("eBGDAE"):
                line = i + ((step - overflow) * 7)
                tab = self.tabs[string][step*maxlen:(step+1)*maxlen]
                screen.addstr(line, 0, "{}: {}".format(string, tab))
        screen.refresh()

    def print_out(self):
        maxlen = self.maxlen - 3
        tablen = len(self.tabs["E"])
        # only most recent line needs updating
        steps = math.ceil(tablen / maxlen)
        for step in range(steps):
            for i,string in enumerate("eBGDAE"):
                tab = self.tabs[string][step*maxlen:(step+1)*maxlen]
                print("{}: {}".format(string, tab))

    def output_to_file(self, filename):
        maxlen = self.maxlen - 3
        tablen = len(self.tabs["E"])
        # only most recent line needs updating
        steps = math.ceil(tablen / maxlen)
        with open(filename, "w") as f:
            for step in range(steps):
                for i,string in enumerate("eBGDAE"):
                    tab = self.tabs[string][step*maxlen:(step+1)*maxlen]
                    f.write("{}: {}\n".format(string, tab))
                f.write("\n")

if __name__ == "__main__":
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()

    notes = [
        {"E": 10},
        {"A": 3, "G": 4, "B": 0},
        {},
        {},
        {"D": 10},
        {"D": 10},
        {"D": 10},
        {},
        {},
    ]

    tabulator = Tabulator()

    try:
        for note in notes:
            tabulator.add_timestep(note)
            tabulator.output_to_curses(stdscr)
            time.sleep(0.2)
    finally:
        time.sleep(1)
        curses.echo()
        curses.nocbreak()
        curses.endwin()

    tabulator.output_to_file("generated_tabs/tab.txt")