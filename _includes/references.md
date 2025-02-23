## References

{% for reference in page.references %}
[{{reference.tag}}]: #{{reference.tag}}
"{{reference.authors}} ({{reference.year}}): {{reference.title}}"
{% endfor %}

<ol>
{% for reference in page.references %}
<li id="{{reference.tag}}">
    {{reference.authors}} ({{reference.year}}):
    <a href="{{reference.url}}"> {{reference.title}} </a>
</li>
{% endfor %}
</ol>
