import curses
import time

class NoteHistory():

    def __init__(self, max_len=20):
        self.strings = {x:[] for x in "EADGBe"}
        self.max_len = max_len
    
    def add_timestep(self, string_fret_map):
        for string,history in self.strings.items():
            if string in string_fret_map:
                formatted = "{:->3}".format(string_fret_map[string])
                # only add note if it was not present in the previous step
                if len(history) == 0 or history[-1] != formatted:
                    history.append(formatted)
                    # don't output the dashes
                    continue
            history.append("---")

    def report(self):
        for i,string in enumerate("eBGDAE"):
            hist = self.strings[string][-self.max_len:]
            stdscr.addstr(i, 0, "{}: {}".format(string, "".join(hist)))
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