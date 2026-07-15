"""Generate an argon2id hash for ROSCRIBE_USERS.

Usage:
  .venv/bin/python scripts/hash_password.py <username>
  (the password is prompted, never echoed or stored)

Put the printed entry in .env. Because argon2 hashes contain commas, join
multiple entries with ';':
  ROSCRIBE_USERS=alice:$argon2id$...;bob:$argon2id$...
"""

from __future__ import annotations

import getpass
import sys

from argon2 import PasswordHasher


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    username = sys.argv[1]
    pw = getpass.getpass(f"Password for {username!r}: ")
    if pw != getpass.getpass("Repeat: "):
        sys.exit("Passwords do not match.")
    print(f"\n{username}:{PasswordHasher().hash(pw)}")


if __name__ == "__main__":
    main()
