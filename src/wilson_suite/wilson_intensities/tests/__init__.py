"""
Test Structure

A good structure for all your tests (this is not limited to unit tests) is this one:

    Set up the test data
    Call your method under test
    Assert that the expected results are returned

There's a nice mnemonic to remember this structure: “Arrange, Act, Assert”.
Another one that you can use takes inspiration from BDD .
It's the “given”, “when”, “then” triad, where given reflects the setup,
when the method call and then the assertion part.

https://martinfowler.com/articles/practical-test-pyramid.html


Defensive Programming

Program defensively, i.e., assume that errors are going to arise, and write code to detect them when they do.
Put assertions in programs to check their state as they run,
    and to help readers understand how those programs are supposed to work.
Use preconditions to check that the inputs to a function are safe to use.
Use postconditions to check that the output from a function is safe to use.
Write tests before writing code in order to help determine exactly what that code is supposed to do.

https://swcarpentry.github.io/python-novice-inflammation/10-defensive.html
"""

from . import testing_utils
