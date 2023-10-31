"""
CSS is the styling used to make node html pretty
when outputting the work a ToT engine does to an
HTML file for debugging.
"""
CSS = """
#wrapper {
    margin: 0 auto;
    width: 80%;
}

details[open] details {
  animation: animateDown 0.2s linear forwards;
}

@keyframes animateDown {
  0% {
    opacity: 0;
    transform: translatey(-15px);
  }
  100% {
    opacity: 1;
    transform: translatey(0);
  }
}

details details {
    margin-left: 20px;
    border-left: 3px rgba(128, 128, 128, 0.5) solid;
    padding-left: 5px;
}

summary {
    display: block;
}

summary::after {
    margin-left: 1ch;
    display: inline-block;
    transition: 0.2s;
}

details summary::after {
    content: '➕';
}

details[open] summary::after {
    content: '➖';
}

details:not([open]) summary::after {
    content: '➕';
}

.complete {
    background-color: rgba(19, 227, 42, 0.5);
}

.invalid {
  background-color: rgba(227, 19, 19, 0.5);
}
"""
