import curses
import time

class NoteHistory():

    def __init__(self, max_len=50):
        self.tabs = {x:"" for x in "EADGBe"}
        self.most_recent_fret = {x:None for x in "EADGBe"}
        self.max_len = max_len
    
    def add_timestep(self, string_fret_map):
        for string,history in self.tabs.items():
            if string in string_fret_map:
                new_fret = string_fret_map[string]
                if self.most_recent_fret[string] != new_fret:
                    self.most_recent_fret[string] = new_fret
                    formatted = "{:->3}".format(string_fret_map[string])
                    self.tabs[string] += formatted
                    # don't add dashes
                    continue
            else:
                self.most_recent_fret[string] = None
            self.tabs[string] += "---"

    def report(self):
        for i,string in enumerate("eBGDAE"):
            stdscr.addstr(i, 0, "{}: {}".format(string, self.tabs[string][-self.max_len:]))
        stdscr.refresh()



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

    N = NoteHistory()

    try:
        for note in notes:
            N.add_timestep(note)
            N.report()
            time.sleep(0.5)
    finally:
        time.sleep(2)
        curses.echo()
        curses.nocbreak()
        curses.endwin()