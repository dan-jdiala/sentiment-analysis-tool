class Reviews:
    """Class to store and manage review data"""

    def __init__(self, username, review):
        """Initialize a review with username and review text"""
        self.username = username
        self.review = review

    def __str__(self):
        """String representation of the review"""
        return f"{self.username}: {self.review}"

    def __repr__(self):
        """Developer-friendly representation"""
        return f"Reviews(username='{self.username}', review='{self.review[:50]}...')"