import argparse
# Main process called by command line
# Main process manages PROGRAMS, programs call submodules for data processing and move data around to achieve a goal.

def flightProgram():
    return
    # Flight program implementation goes here. Outline:
    # Instantiate pipeline
    # Instantiate video mediator, start frame capture, feed tent coordinates into pipeline
    # Feed tent coordinates from pipeline into geolocation
    # Get GPS coordinates from geolocation
    # Send coordinates to command module

def searchProgram():
    return
    # Search program implementation goes here

def taxiProgram():
    return
    # Taxi program implementation goes here

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("program", help="Program name to execute (flight, taxi, search)")
    # Locals is a symbol table, it allows you to execute a function by doing a search of its name.
    locals()[parser.parse_args() + 'Program']() if parser.parse_args() in locals() else None

