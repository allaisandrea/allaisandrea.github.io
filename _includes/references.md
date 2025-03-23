## References

{% for reference in page.references %}
[{{reference.tag}}]: #{{reference.tag}}
"{{reference.authors}} ({{reference.year}}): {{reference.title}}"
{% endfor %}

<ol>
{% for reference in page.references %}
<li id="{{reference.tag}}">
    {{reference.authors}} ({{reference.year}}).
    {% if reference.url %}
        <a href="{{reference.url}}" target='_blank'>{{reference.title}}.</a>
    {% else %}
        {{reference.title}}.
    {% endif %}
    {% if reference.notes %}{{reference.notes}} {% endif %}
</li>
{% endfor %}
</ol>
