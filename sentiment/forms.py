from django import forms

class ReviewForm(forms.Form):
    review = forms.CharField(
        label='Enter your movie review:',
        widget=forms.Textarea(attrs={
            'rows': 5,
            'cols': 60,
            'placeholder': 'For example: This movie was fantastic!'
        }),
        max_length=2000
    )