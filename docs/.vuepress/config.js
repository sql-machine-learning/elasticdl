module.exports = {
  title: 'elasticdl',
  base: '/elasticdl/',
  lang: 'en-US',
  description: 'A Kubernetes-native Deep Learning Framework.',
  themeConfig: {
    repo: 'sql-machine-learning/elasticdl',
    lastUpdated: 'Last Updated',
    editLinks: true,
    docsDir: 'docs',
    nav: [
      { text: 'Guide', link: '/guide/' },
    ],
    sidebar: {
      '/guide/': [
        {
          title: 'Guide',
          collapsable: false,
          children: [
            '',
          ],
        },
      ]
    }
  },
};
