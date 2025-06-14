---
layout: default
title: Welcome
---

# Welcome to My Tech Blog

Hi, I'm Mohamed Rilfi — a software engineer passionate about emerging technologies. My core interests include:

- **Distributed Computing**
- **Big Data Analytics**
- **Deep Learning and AI**
- **Cybersecurity**
- **Parallel Computing & GPU Acceleration**
- **Blockchain Technologies**
- **Solving NP-Hard Problems Using GPU-based Optimization**

I enjoy designing scalable systems, exploring high-performance computing architectures, and tackling complex computational problems. This blog is where I share insights, tutorials, and deep dives into the areas I explore—whether it’s building secure systems or optimizing algorithms for GPU execution.

---

## 📝 Articles by Category

{% assign all_categories = "" | split: "" %}
{% for article in site.articles %}
  {% for cat in article.categories %}
    {% unless all_categories contains cat %}
      {% assign all_categories = all_categories | push: cat %}
    {% endunless %}
  {% endfor %}
{% endfor %}

{% assign sorted_categories = all_categories | sort %}

{% for category in sorted_categories %}
### 📂 {{ category }}
<ul>
  {% for article in site.articles %}
    {% if article.categories contains category %}
      <li><a href="{{ article.url }}">{{ article.title }}</a></li>
    {% endif %}
  {% endfor %}
</ul>
{% endfor %}

---

Stay tuned for more updates!
