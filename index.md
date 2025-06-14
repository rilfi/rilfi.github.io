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

{% assign categorized = site.articles | group_by: "category" | sort: "name" %}

{% for category_group in categorized %}
### 📂 {{ category_group.name | replace: '-', ' ' | capitalize }}
<ul>
  {% assign sorted_items = category_group.items | sort: "title" %}
  {% for article in sorted_items %}
    <li><a href="{{ article.url }}">{{ article.title }}</a></li>
  {% endfor %}
</ul>
{% endfor %}

---

Stay tuned for more updates!
