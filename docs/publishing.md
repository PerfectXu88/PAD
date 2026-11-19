# Publishing the project page

Live site: **https://perfectxu88.github.io/PAD/**

The [Pages workflow](../.github/workflows/pages.yml) publishes
`assets/project-page/` as the website root. The entrypoint is
[`index.html`](../assets/project-page/index.html); styles, scripts, illustrations,
and the line emblem are served alongside it.

## Update the website

Edit the files in `assets/project-page/`, then commit and push to `main`.
The **Deploy PAD project page** workflow automatically publishes those changes.
The workflow can also be started manually from the repository's Actions tab.

Under **Settings → Pages**, the publishing source is **GitHub Actions**.
The workflow uploads only the website directory. Python code, datasets, and
experiment outputs are not part of the deployed website.

## Visual identity

- `assets/project-page/pad-mark.svg`: the website's line emblem.
- `assets/pad-banner.svg`: the matching, self-contained README banner.

The README banner links to the live site. Paper and Code destinations in the
website remain empty until their links are ready to publish.
