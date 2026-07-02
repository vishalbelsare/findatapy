import sys
import datetime

if sys.version_info >= (3, 12):
    def utcnow():
        """Replacement for datetime.datetime.utcnow() that works on Python 3.12+.
        Returns a naive UTC datetime (no tzinfo), same as the old utcnow()."""
        return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
else:
    def utcnow():
        """Returns a naive UTC datetime."""
        return datetime.datetime.utcnow()
